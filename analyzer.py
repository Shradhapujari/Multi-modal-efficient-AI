import whisper
import torch
from transformers import pipeline
from ultralytics import YOLO
import numpy as np
import psutil

# --- Model Identifiers ---
# We use the pre-trained emotion model DIRECTLY. No custom folder needed.
TEXT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
AUDIO_EMOTION_MODEL = "superb/wav2vec2-base-superb-er"
OBJECT_DETECTION_MODEL = "yolov8n.pt"

# --- Threat Definitions ---
# Text: We treat high-arousal negative emotions as 'Harmful Intent' proxies
HARMFUL_TEXT_EMOTIONS = ["anger", "disgust", "fear"]

# Audio: We treat high-arousal negative emotions as 'Risk'
HARMFUL_AUDIO_EMOTIONS = ["angry", "fearful"]

# Video: Objects that confirm a threat
THREAT_OBJECTS = ["knife", "gun", "weapon", "scissors"] 

class ThreatAnalyzer:
    def __init__(self):
        print("Loading Triage Cascade models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. SPEECH-TO-TEXT (Transcription)
        print("Loading Whisper...")
        self.whisper_model = whisper.load_model("base", device=self.device)

        # 2. TEXT MODEL (Tier 1 Gatekeeper)
        # We use the pipeline for instant, easy inference
        print(f"Loading Text Model ({TEXT_EMOTION_MODEL})...")
        self.text_classifier = pipeline("text-classification", model=TEXT_EMOTION_MODEL, device=self.device)

        # 3. AUDIO MODEL (Tier 1 Gatekeeper)
        print(f"Loading Audio Model ({AUDIO_EMOTION_MODEL})...")
        self.audio_classifier = pipeline("audio-classification", model=AUDIO_EMOTION_MODEL, device=self.device)

        # 4. VISION MODEL (Tier 3 Specialist)
        print("Loading YOLO Vision Model...")
        self.yolo_model = YOLO(OBJECT_DETECTION_MODEL)

        print("✅ All Models loaded successfully.")

    def predict_text_intent(self, text):
        """
        Classifies text as Safe (0) or Harmful (1) based on Emotion.
        """
        if not text.strip(): return 0, "neutral"
        
        # The pipeline gives us a list like [{'label': 'anger', 'score': 0.9}]
        result = self.text_classifier(text[:512])[0] 
        label = result['label']
        
        # Map emotion to Intent (1 = Harmful, 0 = Safe)
        if label in HARMFUL_TEXT_EMOTIONS:
            return 1, label
        else:
            return 0, label

    def analyze_chunk(self, audio_chunk, video_frames, samplerate=16000):
        """
        Runs the 'Triage Cascade' logic.
        """
        models_used = {"text": 0, "audio": 0, "vision": 0}
        
        # Normalize audio
        audio_float = audio_chunk.astype(np.float32)
        if np.abs(audio_float).max() > 1.0: audio_float = audio_float / 32768.0

        # ==============================================================================
        # TIER 1: TRIAGE (Run Text & Audio in Parallel)
        # ==============================================================================
        models_used["text"] = 1
        models_used["audio"] = 1
        
        # A. Process Text
        transcription = self.whisper_model.transcribe(audio_chunk.flatten().astype(np.float32), fp16=False)["text"].strip()
        text_intent, text_emotion = self.predict_text_intent(transcription) 
        
        # B. Process Audio
        try:
            emotions = self.audio_classifier(audio_float, sampling_rate=samplerate)
            audio_emotion = emotions[0]['label'] # e.g., 'neu', 'hap', 'ang'
            # Map to risk (1 = Risk, 0 = Safe)
            audio_risk = 1 if audio_emotion in ['ang', 'fea', 'sad'] else 0 
        except:
            audio_emotion = "unknown"
            audio_risk = 0

        print(f"  [Tier 1] Text: '{transcription}' ({text_emotion}) | Audio: {audio_emotion}")

        # ==============================================================================
        # TIER 2: RELIABILITY ENGINE (Logic Check)
        # ==============================================================================
        
        # LOGIC 1: AGREEMENT -> Fast Exit (Efficiency Win)
        if text_intent == audio_risk:
            if text_intent == 0:
                return "SAFE (Context Matches)", models_used
            else:
                return "HARMFUL (Context Matches)", models_used

        # LOGIC 2: MISMATCH -> Ambiguity Detected
        print(f"  [!] MISMATCH (Text={text_emotion}, Audio={audio_emotion}). Escalating...")

        # ==============================================================================
        # TIER 3: SPECIALIST (Vision Fusion)
        # ==============================================================================
        models_used["vision"] = 1
        
        for frame in video_frames:
            if frame is None: continue
            results = self.yolo_model(frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    label = self.yolo_model.names[class_id]
                    if label in THREAT_OBJECTS and box.conf > 0.5:
                        return f"HARMFUL: Visual threat '{label}' confirmed.", models_used
        
        # Fallback behavior
        final_verdict = "HARMFUL" if text_intent == 1 else "SAFE"
        return f"{final_verdict} (Ambiguity Resolved by Vision)", models_used

    def analyze_chunk_baseline(self, audio_chunk, video_frames, samplerate=16000):
        """
        The Baseline: Runs EVERYTHING every time.
        """
        models_used = {"text": 1, "audio": 1, "vision": 1}
        
        audio_float = audio_chunk.astype(np.float32)
        if np.abs(audio_float).max() > 1.0: audio_float = audio_float / 32768.0
        _ = self.whisper_model.transcribe(audio_chunk.flatten().astype(np.float32), fp16=False)["text"]
        _ = self.audio_classifier(audio_float, sampling_rate=samplerate)
        
        for frame in video_frames:
             if frame is None: continue
             _ = self.yolo_model(frame, verbose=False)
        
        return "Baseline analysis complete.", models_used
