import time
import psutil
from stream_handler import StreamHandler
from analyzer import ThreatAnalyzer
import numpy as np
import json
import os
import datetime

# --- Configuration ---
ANALYSIS_DURATION_SECONDS = 60
CHUNK_DURATION_SECONDS = 5
ANALYSIS_MODE = "hierarchical"

# --- NEW: Cold Path Setup ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def log_interaction_to_cold_storage(metrics):
    """
    Simulates sending data to a cold storage path by saving it as a JSON file.
    In a real system, this would upload to S3, a data lake, etc.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(LOG_DIR, f"interaction_chunk_{metrics['chunk']}_{timestamp}.json")
    
    # We don't save the raw audio/video in this simple log to save space,
    # but in a real system, you would save the file path to the raw data.
    log_data = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "analysis_mode": ANALYSIS_MODE,
        "chunk_number": metrics['chunk'],
        "result": metrics['result'],
        "latency_ms": metrics['latency_ms'],
        "cpu_percent": metrics['cpu_percent'],
        "models_used": metrics['models_used']
        # In a real system, add: "audio_file_path": "s3://bucket/...", "user_feedback": "correct/incorrect"
    }
    
    with open(filename, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"  [Cold Path] Logged chunk {metrics['chunk']} data to {filename}")

def main():
    stream = StreamHandler(duration=CHUNK_DURATION_SECONDS)
    analyzer = ThreatAnalyzer()
    time.sleep(2)
    stream.start()

    all_metrics = []
    total_chunks = ANALYSIS_DURATION_SECONDS // CHUNK_DURATION_SECONDS

    print(f"\n--- Starting {ANALYSIS_MODE.upper()} Analysis for {ANALYSIS_DURATION_SECONDS} seconds ---")

    try:
        for i in range(total_chunks):
            start_time = time.time()
            print(f"\n--- Evaluating Chunk {i+1}/{total_chunks} ---")

            audio_chunk, video_frames = stream.read_chunk()
            print(f"DEBUG: Audio chunk shape: {audio_chunk.shape}, dtype: {audio_chunk.dtype}")

            if ANALYSIS_MODE == "hierarchical":
                result, models_used = analyzer.analyze_chunk(audio_chunk, video_frames)
            else:
                result, models_used = analyzer.analyze_chunk_baseline(audio_chunk, video_frames)

            end_time = time.time()
            latency = (end_time - start_time) * 1000
            cpu_usage = psutil.cpu_percent()

            metrics = {
                "chunk": i + 1,
                "result": result,
                "latency_ms": latency,
                "cpu_percent": cpu_usage,
                "models_used": models_used
            }
            all_metrics.append(metrics)
            
            # --- NEW: Trigger the Cold Path ---
            log_interaction_to_cold_storage(metrics)

            print(f"Result: {result}")
            print(f"Metrics: Latency={latency:.2f}ms, CPU Usage={cpu_usage}%")
            print(f"Models Used This Chunk: {models_used}")

    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
    finally:
        stream.stop()
        print("\n--- Analysis Complete ---")
        summarize_results(all_metrics)

def summarize_results(metrics):
    if not metrics:
        print("No data to summarize.")
        return

    avg_latency = np.mean([m['latency_ms'] for m in metrics])
    avg_cpu = np.mean([m['cpu_percent'] for m in metrics])
    
    # --- FIX: Use the correct keys ('audio' instead of 'tone') ---
    total_text_runs = sum(m['models_used'].get('text', 0) for m in metrics)
    total_audio_runs = sum(m['models_used'].get('audio', 0) for m in metrics)
    total_vision_runs = sum(m['models_used'].get('vision', 0) for m in metrics)
    
    print("\n--- Final Summary ---")
    print(f"Analysis Mode: {ANALYSIS_MODE.upper()}")
    print(f"Average Latency per Chunk: {avg_latency:.2f} ms")
    print(f"Average CPU Usage: {avg_cpu:.2f} %")
    print("\nModel Usage Counts:")
    print(f"  - Text Model:   Ran {total_text_runs} times")
    print(f"  - Audio Model:  Ran {total_audio_runs} times")
    print(f"  - Vision Model: Ran {total_vision_runs} times") # This will be lower than text/audio!
    print("\n----------------------\n")


if __name__ == "__main__":
    main()
