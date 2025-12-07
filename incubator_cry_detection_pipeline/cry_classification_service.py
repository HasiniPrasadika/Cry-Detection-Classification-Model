#!/usr/bin/env python3
"""
Cry Classification Service - FastAPI Production Service
Hybrid YAMNet Detection + Ensemble Classification

API Endpoints:
- POST /classify - Classify uploaded audio file
- GET /health - Service health check
- GET /model-info - Model configuration details

Features:
- Two-stage pipeline: YAMNet detection → Ensemble classification
- File upload support (WAV, MP3, M4A)
- Confidence thresholds for quality control
- Caching for model loading efficiency

Usage:
    uvicorn cry_classification_service:app --host 0.0.0.0 --port 8890

Dependencies:
    pip3 install fastapi uvicorn python-multipart tensorflow tensorflow-hub librosa numpy joblib scikit-learn
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import joblib
import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Cry Classification Service",
    description="YAMNet-based cry detection and ensemble classification",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths configuration
BASE_DIR = Path(__file__).parent / "Cry-Detection-Classification-Model" / "cry_project"
DET_MODELS_DIR = BASE_DIR / "det_models"
CLASS_MODELS_DIR = BASE_DIR / "models"

# Detection model paths
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
DET_LR_MODEL_PATH = DET_MODELS_DIR / "yamnet_lr_model.joblib"
DET_SCALER_PATH = DET_MODELS_DIR / "scaler_yamnet.pkl"
DET_PCA_PATH = DET_MODELS_DIR / "pca_yamnet.pkl"

# Classification model paths
CLASS_ENSEMBLE_PATH = CLASS_MODELS_DIR / "babycry_ensemble.pkl"
CLASS_SCALER_PATH = CLASS_MODELS_DIR / "scaler.pkl"
CLASS_SELECTOR_PATH = CLASS_MODELS_DIR / "feature_selector.pkl"
CLASS_LABEL_ENCODER_PATH = CLASS_MODELS_DIR / "label_encoder.pkl"

# Configuration
DETECTION_THRESHOLD = 0.212  # Optimized from training
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.6
AUDIO_SAMPLE_RATE = 16000

# Global model storage
models = {
    "yamnet": None,
    "det_lr_model": None,
    "det_scaler": None,
    "det_pca": None,
    "class_ensemble": None,
    "class_scaler": None,
    "class_selector": None,
    "class_label_encoder": None,
}

class ModelLoadError(Exception):
    """Custom exception for model loading failures"""
    pass

def load_yamnet_model() -> tf.keras.Model:
    """Load YAMNet model from TensorFlow Hub (cached)"""
    try:
        logger.info("Loading YAMNet model from TensorFlow Hub...")
        model = hub.load(YAMNET_MODEL_URL)
        logger.info("✓ YAMNet model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"✗ Failed to load YAMNet model: {e}")
        raise ModelLoadError(f"YAMNet loading failed: {e}")

def load_detection_models() -> Tuple[object, object, object]:
    """Load detection pipeline models (LR + Scaler + PCA)"""
    try:
        logger.info("Loading detection models...")
        lr_model = joblib.load(DET_LR_MODEL_PATH)
        scaler = joblib.load(DET_SCALER_PATH)
        pca = joblib.load(DET_PCA_PATH)
        logger.info("✓ Detection models loaded successfully")
        return lr_model, scaler, pca
    except Exception as e:
        logger.error(f"✗ Failed to load detection models: {e}")
        raise ModelLoadError(f"Detection models loading failed: {e}")

def load_classification_models() -> Tuple[object, object, object, object]:
    """Load classification pipeline models (Ensemble + Scaler + Selector + Label Encoder)"""
    try:
        logger.info("Loading classification models...")
        ensemble = joblib.load(CLASS_ENSEMBLE_PATH)
        scaler = joblib.load(CLASS_SCALER_PATH)
        selector = joblib.load(CLASS_SELECTOR_PATH)
        label_encoder = joblib.load(CLASS_LABEL_ENCODER_PATH)
        logger.info("✓ Classification models loaded successfully")
        return ensemble, scaler, selector, label_encoder
    except Exception as e:
        logger.error(f"✗ Failed to load classification models: {e}")
        raise ModelLoadError(f"Classification models loading failed: {e}")

def extract_yamnet_embeddings(audio_path: str) -> np.ndarray:
    """
    Extract YAMNet embeddings from audio file
    Returns: 2048-dimensional feature vector (mean + std of embeddings)
    """
    try:
        # Load audio file
        waveform, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)
        
        # Convert to float32 tensor
        waveform_tensor = tf.cast(waveform, tf.float32)
        
        # Extract YAMNet embeddings
        scores, embeddings, spectrogram = models["yamnet"](waveform_tensor)
        
        # Aggregate embeddings: [mean(1024), std(1024)] = 2048 dims
        embeddings_np = embeddings.numpy()
        mean_embedding = np.mean(embeddings_np, axis=0)  # 1024 dims
        std_embedding = np.std(embeddings_np, axis=0)    # 1024 dims
        
        # Concatenate to 2048 dims
        aggregated_features = np.concatenate([mean_embedding, std_embedding])
        
        return aggregated_features
        
    except Exception as e:
        logger.error(f"YAMNet embedding extraction failed: {e}")
        raise

def detect_cry(audio_path: str) -> Tuple[bool, float]:
    """
    Stage 1: Detect if audio contains crying
    Returns: (is_cry: bool, confidence: float)
    """
    try:
        # Extract YAMNet embeddings
        features = extract_yamnet_embeddings(audio_path)
        features = features.reshape(1, -1)  # (1, 2048)
        
        # Scale features
        features_scaled = models["det_scaler"].transform(features)
        
        # Apply PCA
        features_pca = models["det_pca"].transform(features_scaled)
        
        # Predict with Logistic Regression
        cry_prob = models["det_lr_model"].predict_proba(features_pca)[0][1]  # Probability of "cry" class
        is_cry = cry_prob >= DETECTION_THRESHOLD
        
        logger.info(f"Detection: is_cry={is_cry}, confidence={cry_prob:.3f}")
        return is_cry, float(cry_prob)
        
    except Exception as e:
        logger.error(f"Cry detection failed: {e}")
        raise

def extract_classification_features(audio_path: str) -> np.ndarray:
    """
    Extract librosa features for classification (matches training pipeline)
    Features: MFCC (40) + Chroma (12) + Mel (40) + Contrast (7) + Tonnetz (6) + 
              Zero Crossing (1) + Energy (1) + Spectral (5) = 111 dims
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)
        
        # Compute STFT for chroma and contrast
        stft = np.abs(librosa.stft(y))
        
        # MFCC (40 coefficients)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
        
        # Chroma (12 bins)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr), axis=1)
        
        # Mel Spectrogram (40 bands)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
        
        # Spectral Contrast (7 bands)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr), axis=1)
        
        # Tonnetz (6 features)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1)
        
        # Zero crossing rate (1 feature)
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Energy / RMS (1 feature)
        energy = np.mean(librosa.feature.rms(y=y))
        
        # Spectral features (5 features)
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spec_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # Combine all features (111 dims total)
        combined_features = np.concatenate([
            mfcc[:40],          # 40
            chroma[:12],        # 12
            mel[:40],           # 40
            contrast[:7],       # 7
            tonnetz[:6],        # 6
            [zero_crossing],    # 1
            [energy],           # 1
            [spec_centroid],    # 1
            [spec_bandwidth],   # 1
            [spec_rolloff],     # 1
            [spec_flatness]     # 1
        ])
        
        return combined_features
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise

def classify_cry(audio_path: str) -> Tuple[str, Dict[str, float], float]:
    """
    Stage 2: Classify cry type
    Returns: (label: str, probabilities: dict, confidence: float)
    """
    try:
        # Extract features
        features = extract_classification_features(audio_path)
        features = features.reshape(1, -1)  # (1, 107)
        
        # Scale features
        features_scaled = models["class_scaler"].transform(features)
        
        # Apply feature selection
        features_selected = models["class_selector"].transform(features_scaled)
        
        # Predict with ensemble
        probabilities = models["class_ensemble"].predict_proba(features_selected)[0]
        predicted_class_idx = np.argmax(probabilities)
        max_prob = float(probabilities[predicted_class_idx])
        
        # Decode label
        label = models["class_label_encoder"].inverse_transform([predicted_class_idx])[0]
        
        # Create probability dictionary
        all_labels = models["class_label_encoder"].classes_
        prob_dict = {label: float(prob) for label, prob in zip(all_labels, probabilities)}
        
        logger.info(f"Classification: label={label}, confidence={max_prob:.3f}")
        return label, prob_dict, max_prob
        
    except Exception as e:
        logger.error(f"Cry classification failed: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load all models on service startup"""
    logger.info("=" * 60)
    logger.info("🍼 Cry Classification Service Starting...")
    logger.info("=" * 60)
    
    try:
        # Verify model paths exist
        logger.info("Checking model paths...")
        required_paths = [
            DET_LR_MODEL_PATH, DET_SCALER_PATH, DET_PCA_PATH,
            CLASS_ENSEMBLE_PATH, CLASS_SCALER_PATH, CLASS_SELECTOR_PATH, CLASS_LABEL_ENCODER_PATH
        ]
        
        for path in required_paths:
            if not path.exists():
                raise ModelLoadError(f"Model file not found: {path}")
        
        logger.info("✓ All model paths verified")
        
        # Load YAMNet
        models["yamnet"] = load_yamnet_model()
        
        # Load detection models
        models["det_lr_model"], models["det_scaler"], models["det_pca"] = load_detection_models()
        
        # Load classification models
        (models["class_ensemble"], models["class_scaler"], 
         models["class_selector"], models["class_label_encoder"]) = load_classification_models()
        
        logger.info("=" * 60)
        logger.info("✅ All models loaded successfully!")
        logger.info(f"📊 Detection threshold: {DETECTION_THRESHOLD}")
        logger.info(f"📊 Classification confidence threshold: {CLASSIFICATION_CONFIDENCE_THRESHOLD}")
        logger.info(f"🎤 Audio sample rate: {AUDIO_SAMPLE_RATE} Hz")
        logger.info(f"🏷️  Cry classes: {list(models['class_label_encoder'].classes_)}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.get("/health")
async def health_check():
    """Service health check endpoint"""
    models_loaded = all(v is not None for v in models.values())
    
    return JSONResponse(
        status_code=200 if models_loaded else 503,
        content={
            "status": "healthy" if models_loaded else "unhealthy",
            "service": "cry-classification-service",
            "version": "1.0.0",
            "models_loaded": models_loaded,
            "timestamp": time.time()
        }
    )

@app.get("/model-info")
async def model_info():
    """Get model configuration details"""
    return JSONResponse(
        content={
            "service": "Cry Classification Service",
            "pipeline": "YAMNet Detection → Ensemble Classification",
            "detection": {
                "model": "YAMNet + LogisticRegression",
                "threshold": DETECTION_THRESHOLD,
                "features": "2048 dims (YAMNet embeddings aggregated)"
            },
            "classification": {
                "model": "Ensemble (Voting Classifier)",
                "confidence_threshold": CLASSIFICATION_CONFIDENCE_THRESHOLD,
                "features": "107 dims (MFCC, Chroma, Mel, Contrast, Tonnetz, Spectral)",
                "classes": list(models["class_label_encoder"].classes_) if models["class_label_encoder"] else []
            },
            "audio": {
                "sample_rate": AUDIO_SAMPLE_RATE,
                "supported_formats": ["wav", "mp3", "m4a", "ogg"]
            }
        }
    )

@app.post("/classify")
async def classify_audio(file: UploadFile = File(...)):
    """
    Classify uploaded audio file
    
    Returns:
    - is_cry: bool (Stage 1 detection result)
    - cry_confidence: float (Detection confidence)
    - classification: str | null (Cry type label, null if not cry or low confidence)
    - classification_confidence: float | null (Classification confidence)
    - probabilities: dict | null (All class probabilities)
    - message: str (Human-readable result)
    """
    temp_path = None
    
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ['wav', 'mp3', 'm4a', 'ogg']:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file_ext}. Supported: wav, mp3, m4a, ogg"
            )
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        logger.info(f"Processing uploaded file: {file.filename} ({len(content)} bytes)")
        
        # Stage 1: Cry Detection
        is_cry, cry_confidence = detect_cry(temp_path)
        
        # Initialize classification results
        classification = None
        classification_confidence = None
        probabilities = None
        message = ""
        
        if not is_cry:
            message = f"No cry detected (confidence: {cry_confidence:.2%})"
            logger.info(message)
        else:
            # Stage 2: Cry Classification
            label, prob_dict, max_prob = classify_cry(temp_path)
            
            # Check confidence threshold
            if max_prob >= CLASSIFICATION_CONFIDENCE_THRESHOLD:
                classification = label
                classification_confidence = max_prob
                probabilities = prob_dict
                message = f"Cry detected: {label} (confidence: {max_prob:.2%})"
                logger.info(message)
            else:
                message = f"Cry detected but classification confidence too low ({max_prob:.2%} < {CLASSIFICATION_CONFIDENCE_THRESHOLD:.2%})"
                logger.warning(message)
        
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "is_cry": bool(is_cry),  # Convert numpy bool to Python bool
                "cry_confidence": round(float(cry_confidence), 4),
                "classification": classification,
                "classification_confidence": round(float(classification_confidence), 4) if classification_confidence else None,
                "probabilities": {k: round(float(v), 4) for k, v in probabilities.items()} if probabilities else None,
                "message": message,
                "timestamp": time.time()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {e}")
        
        # Clean up temp file on error
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint - service info"""
    return {
        "service": "Cry Classification Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /classify": "Classify audio file (multipart/form-data)",
            "GET /health": "Service health check",
            "GET /model-info": "Model configuration details"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8890)
