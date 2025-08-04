#!/usr/bin/env python3
"""Minimal GMM-based speaker clustering"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.enhanced_conversation_db import EnhancedConversationDB

def cluster_speakers(n_speakers=2):
    """Cluster audio features using GMM to identify speakers"""
    db = EnhancedConversationDB()
    features, metadata = db.get_audio_features_for_clustering()
    
    if len(features) < n_speakers:
        return print(f"Need at least {n_speakers} samples, found {len(features)}")
    
    X = StandardScaler().fit_transform(features)
    gmm = GaussianMixture(n_components=n_speakers, random_state=42)
    labels = gmm.fit_predict(X)
    
    print(f"🎤 Found {n_speakers} speakers in {len(features)} samples")
    for i, (meta, label) in enumerate(zip(metadata, labels)):
        speaker_id = chr(65 + label)  # A, B, C...
        print(f"  {meta.get('timestamp', '')[:19]} | Speaker {speaker_id} | {meta.get('speaker', 'Unknown')}")
    
    return labels, metadata

def find_optimal_speakers():
    """Find optimal number of speakers using BIC score"""
    db = EnhancedConversationDB()
    features, metadata = db.get_audio_features_for_clustering()
    
    if len(features) < 4:
        return print(f"Need at least 4 samples for optimization, found {len(features)}")
    
    X = StandardScaler().fit_transform(features)
    max_speakers = min(20, len(features) // 2)
    
    bic_scores = []
    for n in range(1, max_speakers + 1):
        gmm = GaussianMixture(n_components=n, random_state=42)
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
    features, metadata = db.get_audio_features_for_clustering()
    
    if len(features) < 4:
        return print(f"Need at least 4 samples, found {len(features)}")
    
    X = StandardScaler().fit_transform(features)
    scaler = StandardScaler().fit(features)
    max_speakers = min(20, len(features) // 2)
    
    # Find optimal number of speakers
    bic_scores = []
    for n in range(1, max_speakers + 1):
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
    
    optimal_n = np.argmin(bic_scores) + 1
    print(f"🎯 Optimal speakers: {optimal_n} (BIC: {bic_scores[optimal_n-1]:.1f})")
    
    # Train final GMM and update database
    gmm = GaussianMixture(n_components=optimal_n, random_state=42)
    gmm.fit(X)
    
    return db.update_speakers_with_gmm(gmm, scaler, confidence_threshold)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "auto":
        find_optimal_speakers()
    elif len(sys.argv) > 1 and sys.argv[1] == "update":
        confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
        update_database_speakers(confidence)
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
        cluster_speakers(n)