#!/usr/bin/env python3
"""
WORKING Gaming Security Test - No Progress Bar Issues
Fixed version that won't hang at 100%
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
import sys

def safe_progress(current, total, prefix="", suffix="", bar_length=30):
    """Safe progress bar that won't hang"""
    if total <= 0:
        print(f"{prefix} [ERROR: total={total}] {suffix}")
        return
    
    if current > total:
        current = total  # Prevent overflow
    
    percent = min(100.0, (current / total) * 100)
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    # Print on same line
    sys.stdout.write(f'\r{prefix} [{bar}] {percent:.1f}% {suffix}')
    sys.stdout.flush()
    
    # Always print newline when complete
    if current >= total:
        print()

def working_gaming_test():
    """Working gaming security test without hanging"""
    print("🎮 WORKING GAMING SECURITY TEST")
    print("=" * 50)
    
    try:
        # Generate manageable dataset
        print("📊 Generating gaming data (5,000 samples)...")
        np.random.seed(42)
        
        n_samples = 5000  # Smaller, manageable
        n_features = 10
        
        # Generate gaming features
        X = np.random.randn(n_samples, n_features)
        
        # Gaming-specific features
        X[:, 0] = np.random.poisson(2, n_samples)  # kills_per_game
        X[:, 1] = np.random.poisson(1.5, n_samples)  # deaths_per_game
        X[:, 2] = np.random.uniform(0, 30, n_samples)  # playtime_minutes
        X[:, 3] = np.random.uniform(0, 1, n_samples)  # win_rate
        X[:, 4] = np.random.uniform(0, 100, n_samples)  # accuracy_percentage
        
        # Suspicious behavior patterns
        suspicious_score = (
            (X[:, 0] > 8).astype(float) * 0.3 +  # Unusual kills
            (X[:, 4] > 90).astype(float) * 0.3 +  # Perfect accuracy
            (X[:, 2] < 5).astype(float) * 0.2 +  # Very short playtime
            np.random.random(n_samples) * 0.2  # Random noise
        )
        
        # Create binary labels (0 = legitimate, 1 = cheating)
        y = (suspicious_score > 0.4).astype(int)
        
        print(f"✅ Generated {n_samples:,} samples with {n_features} features")
        print(f"📈 Cheating samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model with progress
        print("🤖 Training gaming security model...")
        start_time = time.time()
        
        model = RandomForestClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        
        # Simulate training progress
        epochs = 10
        for epoch in range(epochs):
            safe_progress(epoch + 1, epochs, "Training", f"Epoch {epoch + 1}/{epochs}")
            time.sleep(0.1)  # Simulate work
        
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"⏱️ Training completed in {training_time:.2f} seconds")
        
        # Test model
        print("🧪 Testing model performance...")
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n📊 RESULTS:")
        print("=" * 30)
        print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"⏱️ Training Time: {training_time:.2f}s")
        print(f"📈 Training Samples: {len(X_train):,}")
        print(f"🧪 Test Samples: {len(X_test):,}")
        
        # Feature importance
        feature_names = ['kills', 'deaths', 'playtime', 'win_rate', 'accuracy', 
                       'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10']
        
        importances = model.feature_importances_
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
        
        print("\n🔍 TOP 5 IMPORTANT FEATURES:")
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"{i}. {feature}: {importance:.4f}")
        
        # Test with sample data
        print("\n🎮 SAMPLE PREDICTIONS:")
        print("=" * 30)
        
        # Legitimate player
        legit_player = np.array([[2, 1, 20, 0.45, 65, 0.1, 0.2, 0.3, 0.1, 0.2]])
        legit_pred = model.predict_proba(legit_player)[0]
        print(f"👤 Legitimate Player: {legit_pred[1]:.3f} cheating probability")
        
        # Suspicious player
        sus_player = np.array([[12, 0, 2, 0.98, 98, 0.9, 0.8, 0.9, 0.7, 0.8]])
        sus_pred = model.predict_proba(sus_player)[0]
        print(f"🚨 Suspicious Player: {sus_pred[1]:.3f} cheating probability")
        
        return {
            'accuracy': accuracy,
            'training_time': training_time,
            'samples': n_samples,
            'features': n_features,
            'model': model,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    print("🚀 STARTING WORKING GAMING TEST...")
    print("This version has fixed progress bar logic and won't hang!")
    print()
    
    results = working_gaming_test()
    
    if results.get('success', False):
        print(f"\n🎉 TEST COMPLETED SUCCESSFULLY!")
        print(f"✅ Achieved {results['accuracy']*100:.2f}% accuracy")
        print(f"⚡ Trained on {results['samples']:,} samples")
        print(f"🔧 Used {results['features']} features")
        print(f"⏱️ Training time: {results['training_time']:.2f}s")
    else:
        print(f"\n❌ TEST FAILED!")
        print(f"Error: {results.get('error', 'Unknown error')}")
    
    print("\n🎯 Test completed without hanging!")
