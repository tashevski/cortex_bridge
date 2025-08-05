#!/usr/bin/env python3
"""Minimal GMM-based speaker clustering"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.enhanced_conversation_db import EnhancedConversationDB

def filter_consistent_dimensions(features, metadata):
    """Filter features to most common dimension size for GMM clustering"""
    if not features:
        return [], []
    
    dimensions = [len(f) for f in features]
    most_common_dim = max(set(dimensions), key=dimensions.count)
    
    filtered_features = []
    filtered_metadata = []
    for feat, meta in zip(features, metadata):
        if len(feat) == most_common_dim:
            filtered_features.append(feat)
            filtered_metadata.append(meta)
    
    print(f"📊 Using {len(filtered_features)}/{len(features)} samples with {most_common_dim}D features")
    return filtered_features, filtered_metadata

def cluster_speakers(n_speakers=2):
    """Cluster audio features using GMM to identify speakers"""
    db = EnhancedConversationDB()
    features, metadata = db.get_data("audio_features", return_features=True)
    
    # Filter to consistent dimensions for GMM
    features, metadata = filter_consistent_dimensions(features, metadata)
    
    if len(features) < n_speakers:
        return print(f"Need at least {n_speakers} samples, found {len(features)}")
    
    X = StandardScaler().fit_transform(features)
    gmm = GaussianMixture(n_components=n_speakers, covariance_type='diag', reg_covar=1e-4, random_state=42)
    labels = gmm.fit_predict(X)
    
    print(f"🎤 Found {n_speakers} speakers in {len(features)} samples")
    for i, (meta, label) in enumerate(zip(metadata, labels)):
        speaker_id = chr(65 + label)  # A, B, C...
        print(f"  {meta.get('timestamp', '')[:19]} | Speaker {speaker_id} | {meta.get('speaker', 'Unknown')}")
    
    return labels, metadata

def find_optimal_speakers():
    """Find optimal number of speakers using BIC score"""
    db = EnhancedConversationDB()
    features, metadata = db.get_data("audio_features", return_features=True)
    
    # Filter to consistent dimensions for GMM
    features, metadata = filter_consistent_dimensions(features, metadata)
    
    if len(features) < 4:
        return print(f"Need at least 4 samples for optimization, found {len(features)}")
    
    X = StandardScaler().fit_transform(features)
    max_speakers = min(20, len(features) // 2)
    
    bic_scores = []
    for n in range(1, max_speakers + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='diag', reg_covar=1e-4, random_state=42)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
    
    optimal_n = np.argmin(bic_scores) + 1
    best_bic = bic_scores[optimal_n - 1]
    
    print(f"🎯 Optimal speakers: {optimal_n} (BIC: {best_bic:.1f})")
    print(f"📊 Tested {max_speakers} configurations")
    
    # Run clustering with optimal number
    labels, _ = cluster_speakers(optimal_n)
    return optimal_n, labels, metadata

def update_database_speakers(confidence_threshold=0.8):
    """Find optimal speakers and update database with GMM results"""
    db = EnhancedConversationDB()
    all_features, all_metadata = db.get_data("audio_features", return_features=True)
    
    # Filter to consistent dimensions for GMM
    features, metadata = filter_consistent_dimensions(all_features, all_metadata)
    
    if len(features) < 4:
        return print(f"Need at least 4 samples, found {len(features)}")
    
    X = StandardScaler().fit_transform(features)
    scaler = StandardScaler().fit(features)
    max_speakers = min(20, len(features) // 2)
    
    # Find optimal number of speakers
    bic_scores = []
    for n in range(1, max_speakers + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='diag', reg_covar=1e-4, random_state=42)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
    
    optimal_n = np.argmin(bic_scores) + 1
    print(f"🎯 Optimal speakers: {optimal_n} (BIC: {bic_scores[optimal_n-1]:.1f})")
    
    # Train final GMM and save model
    gmm = GaussianMixture(n_components=optimal_n, covariance_type='diag', reg_covar=1e-4, random_state=42)
    gmm.fit(X)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(base_dir, "models", "speaker_gmm")
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, "gmm_model.pkl"), "wb") as f:
        pickle.dump({'gmm': gmm, 'scaler': scaler}, f)
    
    # Create updates dictionary - map filtered indexes back to original indexes
    updates_dict = {}
    filtered_to_original = {i: all_metadata.index(meta) for i, meta in enumerate(metadata)}
    
    for i, (feature, meta) in enumerate(zip(features, metadata)):
        feature_scaled = scaler.transform([feature])[0]
        probs = gmm.predict_proba([feature_scaled])[0]
        label = gmm.predict([feature_scaled])[0]
        max_prob = np.max(probs)
        
        if max_prob >= confidence_threshold:
            speaker_id = chr(65 + label)  # A, B, C...
            original_index = filtered_to_original[i]
            updates_dict[original_index] = {'ml_speaker': speaker_id, 'ml_speaker_confidence': float(max_prob)}
    
    # Update database using index-based updates
    updated_count = db.update_by_indexes(updates_dict, "audio_features")
    print(f"🎯 Updated {updated_count}/{len(all_features)} speakers (confidence ≥ {confidence_threshold})")
    return updated_count

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "auto":
        find_optimal_speakers()
    elif len(sys.argv) > 1 and sys.argv[1] == "update":
        confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
        update_database_speakers(confidence)
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
        cluster_speakers(n)