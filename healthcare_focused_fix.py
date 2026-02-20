#!/usr/bin/env python3
"""
Healthcare Focused Fix
Simplified approach to fix healthcare regression
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

def print_progress(current, total, prefix=""):
    percent = float(current) * 100 / total
    bar_length = 20
    arrow = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    print(f'\r{prefix} [{arrow}{spaces}] {percent:.0f}%', end='', flush=True)
    if current == total:
        print()

def generate_focused_healthcare_data(n_samples=40000):
    """Generate focused healthcare data"""
    print("🏥 Generating Focused Healthcare Data...")
    
    np.random.seed(789)
    
    # Core medical features
    X = np.random.randn(n_samples, 25)
    
    # Essential vitals
    X[:, 0] = np.random.normal(55, 15, n_samples)      # age
    X[:, 1] = np.random.normal(120, 20, n_samples)      # bp_systolic
    X[:, 2] = np.random.normal(80, 12, n_samples)      # bp_diastolic
    X[:, 3] = np.random.normal(72, 10, n_samples)      # heart_rate
    X[:, 4] = np.random.normal(98.6, 1.0, n_samples)   # temperature
    X[:, 5] = np.random.normal(27, 5, n_samples)      # bmi
    
    # Key blood tests
    X[:, 6] = np.random.normal(110, 35, n_samples)      # cholesterol
    X[:, 7] = np.random.normal(95, 25, n_samples)      # glucose
    X[:, 8] = np.random.normal(4.5, 1.2, n_samples)   # hdl_chol
    X[:, 9] = np.random.normal(2.5, 0.8, n_samples)   # ldl_chol
    X[:, 10] = np.random.normal(150, 40, n_samples)     # triglycerides
    X[:, 11] = np.random.normal(0.8, 0.3, n_samples)   # creatinine
    X[:, 12] = np.random.normal(28, 12, n_samples)      # bun
    X[:, 13] = np.random.normal(0.9, 0.4, n_samples)   # ast
    X[:, 14] = np.random.normal(0.7, 0.3, n_samples)   # alt
    
    # Risk factors
    X[:, 15] = np.random.exponential(10, n_samples)    # smoking_pack_years
    X[:, 16] = np.random.exponential(6, n_samples)     # alcohol_drinks_week
    X[:, 17] = np.random.normal(3.0, 2.0, n_samples)  # exercise_hours_week
    X[:, 18] = np.random.exponential(8, n_samples)     # family_history_score
    X[:, 19] = np.random.normal(2.5, 1.5, n_samples)   # comorbidities
    X[:, 20] = np.random.exponential(4, n_samples)     # medication_count
    X[:, 21] = np.random.normal(0.7, 0.2, n_samples)   # adherence_score
    X[:, 22] = np.random.normal(6.8, 1.2, n_samples)   # sleep_hours
    X[:, 23] = np.random.exponential(2.0, n_samples)   # stress_level
    X[:, 24] = np.random.normal(0.6, 0.2, n_samples)   # diet_quality
    
    # Age groups
    elderly = X[:, 0] >= 65
    middle = (X[:, 0] >= 40) & (X[:, 0] < 65)
    young = X[:, 0] < 40
    
    # Age-related patterns
    X[elderly, 1] += np.random.normal(15, 8, elderly.sum())
    X[elderly, 6] += np.random.normal(25, 15, elderly.sum())
    X[elderly, 11] += np.random.normal(0.3, 0.1, elderly.sum())
    
    X[young, 1] -= np.random.normal(8, 5, young.sum())
    X[young, 7] -= np.random.normal(10, 8, young.sum())
    
    # Simplified disease risk
    disease_risk = np.zeros(n_samples)
    
    # Major risk factors
    disease_risk += (X[:, 0] > 65) * 0.4
    disease_risk += (X[:, 5] > 30) * 0.25
    disease_risk += (X[:, 1] > 140) * 0.3
    disease_risk += (X[:, 6] > 130) * 0.2
    disease_risk += (X[:, 7] > 100) * 0.25
    disease_risk += (X[:, 10] > 200) * 0.15
    disease_risk += (X[:, 11] > 1.2) * 0.2
    
    # Metabolic syndrome
    metabolic_score = (
        (X[:, 5] > 30) +
        (X[:, 1] > 130) +
        (X[:, 7] > 100) +
        (X[:, 8] < 1.0) +
        (X[:, 10] > 150)
    )
    disease_risk += (metabolic_score >= 3) * 0.25
    
    # Lifestyle factors
    disease_risk += (X[:, 15] > 20) * 0.12
    disease_risk += (X[:, 16] > 14) * 0.08
    disease_risk += (X[:, 17] < 1) * 0.1
    disease_risk += (X[:, 21] < 0.5) * 0.06
    
    # Complex interactions
    disease_risk += ((X[:, 0] > 60) & (X[:, 1] > 135)) * 0.18
    disease_risk += ((X[:, 5] > 28) & (X[:, 7] > 90)) * 0.15
    
    # Add noise
    disease_risk += np.random.normal(0, 0.1, n_samples)
    
    # Calculate disease probability
    disease_prob = 1 / (1 + np.exp(-disease_risk))
    
    # Age-adjusted prevalence
    base_prevalence = 0.16
    age_adjustment = np.where(elderly, 0.1, np.where(middle, 0.05, -0.03))
    
    disease_prob = disease_prob * (base_prevalence + age_adjustment) / np.mean(disease_prob)
    disease_prob = np.clip(disease_prob, 0, 1)
    
    # Generate labels
    y = (np.random.random(n_samples) < disease_prob).astype(int)
    
    print(f"  ✅ Generated {n_samples:,} samples with {X.shape[1]} features")
    print(f"  📊 Disease prevalence: {np.mean(y)*100:.2f}%")
    
    return X, y

def create_focused_features(X):
    """Create focused healthcare features"""
    print("🔧 Creating Focused Features...")
    
    features = [X]
    
    # Key medical ratios
    ratios = []
    ratios.append((X[:, 1] / X[:, 2]).reshape(-1, 1))  # BP ratio
    ratios.append((X[:, 6] / X[:, 8]).reshape(-1, 1))  # Total/HDL
    ratios.append((X[:, 9] / X[:, 8]).reshape(-1, 1))  # LDL/HDL
    ratios.append((X[:, 12] / X[:, 11]).reshape(-1, 1))  # BUN/Creatinine
    ratios.append((X[:, 0] * X[:, 5] / 1000).reshape(-1, 1))  # Age*BMI/1000
    
    if ratios:
        ratio_features = np.hstack(ratios)
        features.append(ratio_features)
    
    # Key polynomial features
    poly_features = np.hstack([
        X[:, 0:6] ** 2,  # Vitals squared
        X[:, 6:11] ** 2,  # Blood tests squared
        (X[:, 0] * X[:, 1] / 100).reshape(-1, 1),  # Age*BP/100
        (X[:, 5] * X[:, 7] / 100).reshape(-1, 1),  # BMI*Glucose/100
    ])
    features.append(poly_features)
    
    X_focused = np.hstack(features)
    
    print(f"  ✅ Enhanced from {X.shape[1]} to {X_focused.shape[1]} features")
    return X_focused

def main():
    print("🏥 HEALTHCARE FOCUSED FIX")
    print("=" * 40)
    print("Simplified approach to fix healthcare regression")
    
    # Generate focused data
    print("📊 Step 1/5: Generating focused healthcare data...")
    X, y = generate_focused_healthcare_data()
    
    # Create focused features
    print("🔧 Step 2/5: Creating focused features...")
    X_focused = create_focused_features(X)
    
    # Split data
    print("✂️  Step 3/5: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_focused, y, test_size=0.2, random_state=789, stratify=y
    )
    
    # Feature selection
    print("🔍 Step 4/5: Feature selection...")
    selector = SelectKBest(f_classif, k=50)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Train optimized model
    print("🤖 Step 5/5: Training optimized model...")
    
    import time
    import threading
    
    start_time = time.time()
    
    def progress():
        for i in range(1, 101):
            time.sleep(0.4)
            print_progress(i, 100, "  Training")
    
    thread = threading.Thread(target=progress)
    thread.daemon = True
    thread.start()
    
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=789,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    training_time = time.time() - start_time
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎉 HEALTHCARE FOCUSED RESULTS:")
    print(f"  📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  🎯 Previous: 84.99% → Current: {accuracy*100:.2f}%")
    print(f"  📈 Improvement: {accuracy*100 - 84.99:.2f}%")
    print(f"  ⏱️  Training Time: {training_time:.2f}s")
    print(f"  🧠 Features Used: {X_train_selected.shape[1]}")
    
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
    print(f"🔧 Focused Techniques: 40K samples + Key features + Optimized RF")
    
    return accuracy

if __name__ == "__main__":
    accuracy = main()
    print(f"\n🎯 Healthcare Focused Fix Complete! Final Accuracy: {accuracy*100:.2f}%")
