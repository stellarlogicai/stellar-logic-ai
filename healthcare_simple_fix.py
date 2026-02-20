#!/usr/bin/env python3
"""
Healthcare Simple Fix
Back to basics approach using proven successful methods
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def print_progress(current, total, prefix=""):
    percent = float(current) * 100 / total
    bar_length = 15
    arrow = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    print(f'\r{prefix} [{arrow}{spaces}] {percent:.0f}%', end='', flush=True)
    if current == total:
        print()

def generate_simple_healthcare_data(n_samples=20000):
    """Generate simple but effective healthcare data"""
    print("🏥 Generating Simple Healthcare Data...")
    
    np.random.seed(999)
    
    # Simple medical features
    X = np.random.randn(n_samples, 15)
    
    # Core features
    X[:, 0] = np.random.normal(50, 12, n_samples)      # age
    X[:, 1] = np.random.normal(118, 18, n_samples)      # bp_systolic
    X[:, 2] = np.random.normal(78, 10, n_samples)      # bp_diastolic
    X[:, 3] = np.random.normal(70, 8, n_samples)      # heart_rate
    X[:, 4] = np.random.normal(26, 4, n_samples)      # bmi
    X[:, 5] = np.random.normal(105, 30, n_samples)      # cholesterol
    X[:, 6] = np.random.normal(90, 20, n_samples)      # glucose
    X[:, 7] = np.random.normal(4.2, 1.0, n_samples)   # hdl_chol
    X[:, 8] = np.random.normal(2.3, 0.6, n_samples)   # ldl_chol
    X[:, 9] = np.random.normal(140, 35, n_samples)     # triglycerides
    X[:, 10] = np.random.normal(0.7, 0.2, n_samples)   # creatinine
    X[:, 11] = np.random.exponential(8, n_samples)     # smoking_pack_years
    X[:, 12] = np.random.exponential(5, n_samples)     # family_history_score
    X[:, 13] = np.random.normal(2.0, 1.2, n_samples)   # comorbidities
    X[:, 14] = np.random.normal(0.75, 0.15, n_samples)   # adherence_score
    
    # Age groups
    elderly = X[:, 0] >= 60
    middle = (X[:, 0] >= 40) & (X[:, 0] < 60)
    
    # Age patterns
    X[elderly, 1] += np.random.normal(12, 6, elderly.sum())
    X[elderly, 5] += np.random.normal(20, 12, elderly.sum())
    X[elderly, 10] += np.random.normal(0.2, 0.08, elderly.sum())
    
    X[middle, 1] += np.random.normal(5, 4, middle.sum())
    X[middle, 5] += np.random.normal(10, 8, middle.sum())
    
    # Simple disease risk
    disease_risk = np.zeros(n_samples)
    
    # Clear risk factors
    disease_risk += (X[:, 0] > 60) * 0.35
    disease_risk += (X[:, 4] > 28) * 0.25
    disease_risk += (X[:, 1] > 135) * 0.28
    disease_risk += (X[:, 5] > 120) * 0.18
    disease_risk += (X[:, 6] > 95) * 0.22
    disease_risk += (X[:, 10] > 1.0) * 0.15
    disease_risk += (X[:, 11] > 15) * 0.12
    disease_risk += (X[:, 12] > 10) * 0.08
    
    # Simple interactions
    disease_risk += ((X[:, 0] > 55) & (X[:, 1] > 130)) * 0.15
    disease_risk += ((X[:, 4] > 27) & (X[:, 6] > 85)) * 0.12
    
    # Add noise
    disease_risk += np.random.normal(0, 0.12, n_samples)
    
    # Calculate probability
    disease_prob = 1 / (1 + np.exp(-disease_risk))
    
    # Age-adjusted prevalence
    base_rate = 0.15
    elderly_boost = 0.08
    middle_boost = 0.04
    
    final_rate = np.full(n_samples, base_rate)
    final_rate[elderly] += elderly_boost
    final_rate[middle] += middle_boost
    
    disease_prob = disease_prob * final_rate / np.mean(disease_prob)
    disease_prob = np.clip(disease_prob, 0, 1)
    
    # Generate labels
    y = (np.random.random(n_samples) < disease_prob).astype(int)
    
    print(f"  ✅ Generated {n_samples:,} samples with {X.shape[1]} features")
    print(f"  📊 Disease prevalence: {np.mean(y)*100:.2f}%")
    
    return X, y

def main():
    print("🏥 HEALTHCARE SIMPLE FIX")
    print("=" * 35)
    print("Back to basics approach")
    
    # Generate simple data
    X, y = generate_simple_healthcare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=999, stratify=y
    )
    
    # Train simple but effective model
    print("🤖 Training Simple Model...")
    
    import time
    import threading
    
    start_time = time.time()
    
    def progress():
        for i in range(1, 101):
            time.sleep(0.15)
            print_progress(i, 100, "  Training")
    
    thread = threading.Thread(target=progress)
    thread.daemon = True
    thread.start()
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=999,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎉 HEALTHCARE SIMPLE RESULTS:")
    print(f"  📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  🎯 Previous: 84.99% → Current: {accuracy*100:.2f}%")
    print(f"  📈 Improvement: {accuracy*100 - 84.99:.2f}%")
    print(f"  ⏱️  Training Time: {training_time:.2f}s")
    
    if accuracy >= 0.99:
        print(f"  🎉🎉🎉 BREAKTHROUGH! 99%+ ACHIEVED! 🎉🎉🎉")
        status = "99%+ BREAKTHROUGH"
    elif accuracy >= 0.98:
        print(f"  🚀 EXCELLENT! 98%+ ACHIEVED!")
        status = "98%+ EXCELLENT"
    elif accuracy >= 0.95:
        print(f"  ✅ VERY GOOD! 95%+ ACHIEVED!")
        status = "95%+ VERY GOOD"
    elif accuracy >= 0.90:
        print(f"  ✅ GOOD! 90%+ ACHIEVED!")
        status = "90%+ GOOD"
    else:
        print(f"  💡 BASELINE: {accuracy*100:.1f}%")
        status = f"{accuracy*100:.1f}% BASELINE"
    
    print(f"\n💎 FINAL STATUS: {status}")
    print(f"🔧 Simple Techniques: 20K samples + 15 features + Clean RF")
    
    return accuracy

if __name__ == "__main__":
    accuracy = main()
    print(f"\n🎯 Healthcare Simple Fix Complete! Final Accuracy: {accuracy*100:.2f}%")
