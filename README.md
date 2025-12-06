# Efficient Multimodal Intent Detection for the Visually Impaired using a Triage Cascade Architecture

## 📌 Project Overview
This project presents a novel "Triage Cascade" architecture for real-time, multimodal intent detection designed specifically for wearable assistive devices (e.g., smart glasses) for the visually impaired.

The system detects "Harmful" versus "Safe" intent in social interactions by analyzing three modalities: Text (transcribed speech), Audio (vocal tonality), and Video (facial expressions/body language).

### The Core Problem
Current "State-of-the-Art" Multimodal Fusion models are computationally expensive and battery-hungry, making them impractical for continuous use on wearable devices. Simple cascaded models (Text 
→
 Audio) are efficient but unreliable because they fail to detect context gaps (e.g., a polite phrase spoken in a threatening tone).

### Our Solution: The Triage Cascade
We propose a hybrid architecture that balances high reliability with high efficiency:

Tier 1 (The Triage Layer): Runs lightweight Text and Audio models in parallel to filter clear-cut cases.
Tier 2 (Reliability Engine): Checks for contextual consistency. If signals match (e.g., Safe Text + Safe Audio), it outputs a decision immediately.
Tier 3 (The Specialist): Only escalates to the expensive Video/Fusion model if a Context Mismatch or ambiguity is detected.

## 🏗️ System Architecture
Tier	Function	Models Used	Execution Frequency
Tier 1	Gatekeeper	Text: DistilRoBERTa (Intent)
Audio: Wav2Vec2 (Emotion)	100% of interactions
Tier 2	Logic Check	Rule-based Reliability Engine	100% of interactions
Tier 3	Specialist	Vision: YOLOv8 / Fusion Model	< 20% (Only on escalation)

## 📊 Datasets & Data Engineering
To ensure the model is context-aware and reliable, we engineered custom datasets and applied specific balancing techniques.

1. Text Modality (Intent Detection)
Sources:
Jigsaw Toxicity: For explicit threats (threat, insult, toxic).
GoEmotions: For harmful emotions (anger, fear, disgust).
Better Daily Dialog: For a massive corpus of safe, everyday conversations.
Engineering: The raw data was heavily imbalanced (~90% Safe). We applied Random Oversampling to balance the training set (50/50), which improved Recall on real-world threats by 12%.
2. Audio Modality (Emotion Detection)
Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech).
Focus: Classifying high-arousal emotions (Anger, Fear) as potential risks.
3. Video Modality (Visual Context)
Dataset: FER-2013 (Facial Emotion Recognition) and YOLO pre-trained objects.

## 🧠 Models
Text Intent Model: distilroberta-base fine-tuned on our custom Balanced Master Dataset.
Audio Emotion Model: wav2vec2-base-superb-er (Pre-trained).
Object Detection: yolov8n (Nano) for detecting weapons/threats.
Speech-to-Text: openai/whisper-base.en.

## 📈 Results & Performance
We conducted A/B testing on a Real-World Imbalanced Test Set (15% Harmful / 85% Safe) to validate the system.

1. Reliability (Safety)
Metric	Model A (Class Weights)	Model B (Oversampling)	Improvement
Recall (Harmful)	86.4%	98.5%	+12.1%
Precision	37.6%	48.0%	+10.4%
Missed Threats	591	64	-89% (Critical Win)
2. Efficiency (Battery Life)
Scenario	Active Tiers	Avg. Latency	Avg. CPU Load
Safe / Clear	Tier 1 + Tier 2	~6.3s	~34%
Ambiguous	Tier 1 + 2 + 3	~15.4s	~71%
Conclusion: The Triage Cascade reduces average computational load by ~50% compared to a brute-force baseline while maintaining SOTA reliability.

## 🛠️ Installation & Usage
Prerequisites
Python 3.8+
FFmpeg (Required for Whisper)
1. Clone the Repository
git clone https://github.com/yourusername/intent-detection-cascade.git
cd intent-detection-cascade
2. Install Dependencies
pip install torch torchvision torchaudio transformers ultralytics openai-whisper sounddevice opencv-python psutil librosa numpy scipy scikit-learn datasets accelerate
3. Run the Application
python app/main.py

## 📂 Project Structure
.
├── app/
│   ├── analyzer.py       # Core Triage Cascade logic (Tier 1, 2, 3)
│   ├── main.py           # Main execution loop and logging
│   ├── stream_handler.py # Audio/Video capture handling
│   └── ffmpeg.exe        # (Windows only) Audio processing tool
├── logs/                 # Stores JSON logs of every interaction
├── harmful_safe_distilroberta_OVERSAMPLED/ # The trained Text Model
└── README.md
