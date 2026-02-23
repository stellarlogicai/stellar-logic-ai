#!/usr/bin/env python3
"""
Quick Gaming Security Test - No Progress Bar Issues
Test the gaming security system with smaller dataset
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

def quick_gaming_test():
    """Quick test of gaming security without progress bar issues"""
    print("🎮 QUICK GAMING SECURITY TEST")
    print("=" * 50)
    
    # Generate smaller, manageable dataset
    print("📊 Generating gaming data (10,000 samples)...")
    np.random.seed(42)
    
    n_samples = 10000
    n_features = 15  # Reduced features
    
    # Generate gaming features
    X = np.random.randn(n_samples, n_features)
    
    # Gaming-specific features
    X[:, 0] = np.random.poisson(3, n_samples)  # kills_per_game
    X[:, 1] = np.random.poisson(2, n_samples)  # deaths_per_game
    X[:, 2] = np.random.uniform(0, 60, n_samples)  # playtime_minutes
    X[:, 3] = np.random.uniform(0, 1, n_samples)  # win_rate
    X[:, 4] = np.random.uniform(0, 100, n_samples)  # accuracy_percentage
    
    # Suspicious behavior patterns
    suspicious_score = (
        (X[:, 0] > 10).astype(float) * 0.3 +  # Unusual kills
        (X[:, 4] > 95).astype(float) * 0.3 +  # Perfect accuracy
        (X[:, 2] < 5).astype(float) * 0.2 +  # Very short playtime
        np.random.random(n_samples) * 0.2  # Random noise
    )
    
    # Create binary labels (0 = legitimate, 1 = cheating)
    y = (suspicious_score > 0.5).astype(int)
    
    print(f"✅ Generated {n_samples:,} samples with {n_features} features")
    print(f"📈 Cheating samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("🤖 Training gaming security model...")
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=100,  # Reduced for speed
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
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
                   'feature_6', 'feature_7', 'feature_8', 'feature_9',
                   'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15']
    
    importances = model.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
    
    print("\n🔍 TOP 5 IMPORTANT FEATURES:")
    for i, (feature, importance) in enumerate(top_features, 1):
        print(f"{i}. {feature}: {importance:.4f}")
    
    # Test with sample data
    print("\n🎮 SAMPLE PREDICTIONS:")
    print("=" * 30)
    
    # Legitimate player
    legit_player = np.array([[3, 2, 25, 0.45, 65, 0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2]])
    legit_pred = model.predict_proba(legit_player)[0]
    print(f"👤 Legitimate Player: {legit_pred[1]:.3f} cheating probability")
    
    # Suspicious player
    sus_player = np.array([[15, 1, 3, 0.98, 99, 0.9, 0.8, 0.9, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.8]])
    sus_pred = model.predict_proba(sus_player)[0]
    print(f"🚨 Suspicious Player: {sus_pred[1]:.3f} cheating probability")
    
    return {
        'accuracy': accuracy,
        'training_time': training_time,
        'samples': n_samples,
        'features': n_features,
        'model': model
    }

if __name__ == "__main__":
    results = quick_gaming_test()
    
    print(f"\n🎉 TEST COMPLETED!")
    print(f"✅ Achieved {results['accuracy']*100:.2f}% accuracy")
    print(f"⚡ Trained on {results['samples']:,} samples")
    print(f"🔧 Used {results['features']} features")
    print(f"⏱️ Training time: {results['training_time']:.2f}s")
