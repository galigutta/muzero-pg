import ray
import time
from collections import defaultdict

# Initialize Ray
ray.init()

# Define a remote function for workers to process data
@ray.remote
def worker_task(data):
    # time.sleep(1)  # Simulate processing time
    return data * 2

# Define an analysis function to process consolidated data
def analyze_data(consolidated_data):
    print("Analysis: ", sorted(consolidated_data))

# Generate a stream of data
def data_stream():
    data = 0
    while True:
        time.sleep(.1)  # Simulate data ingestion delay
        data += 1
        yield data

# Shared storage
shared_storage = defaultdict(int)

# Main function to run the distributed program
def main():
    try:
        data_gen = data_stream()
        pending_tasks = []
        
        while True:
            # Ingest data from stream
            data = next(data_gen)

            # Create a worker task with the data and store its ObjectRef
            task = worker_task.remote(data)
            pending_tasks.append(task)

            # Try to get finished tasks and store their results in shared storage
            finished_tasks, pending_tasks = ray.wait(pending_tasks, num_returns=1, timeout=0)
            for task in finished_tasks:
                result = ray.get(task)
                shared_storage[result] += 1

            # Analyze consolidated data
            analyze_data(shared_storage)
            # break
        
    except KeyboardInterrupt:
        print("Terminating program...")

if __name__ == "__main__":
    main()
