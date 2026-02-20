#!/usr/bin/env python3
"""
Healthcare Fast Medical AI
Simplified medical AI approach to avoid hanging
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
    bar_length = 15
    arrow = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    print(f'\r{prefix} [{arrow}{spaces}] {percent:.0f}%', end='', flush=True)
    if current == total:
        print()

def generate_fast_medical_data(n_samples=30000):
    """Generate fast medical AI data"""
    print("🏥 Generating FAST Medical Data...")
    
    np.random.seed(888)
    
    # Focused medical features
    X = np.random.randn(n_samples, 35)
    
    # Core medical features
    X[:, 0] = np.random.normal(52, 16, n_samples)      # age
    X[:, 1] = np.random.normal(122, 22, n_samples)      # bp_systolic
    X[:, 2] = np.random.normal(81, 13, n_samples)      # bp_diastolic
    X[:, 3] = np.random.normal(74, 11, n_samples)      # heart_rate
    X[:, 4] = np.random.normal(98.6, 1.4, n_samples)   # temperature
    X[:, 5] = np.random.normal(27, 6, n_samples)      # bmi
    X[:, 6] = np.random.normal(116, 38, n_samples)     # cholesterol
    X[:, 7] = np.random.normal(94, 26, n_samples)     # glucose
    X[:, 8] = np.random.normal(4.6, 1.3, n_samples)  # hdl_chol
    X[:, 9] = np.random.normal(2.6, 0.9, n_samples)  # ldl_chol
    X[:, 10] = np.random.normal(152, 45, n_samples)    # triglycerides
    X[:, 11] = np.random.normal(0.82, 0.35, n_samples) # creatinine
    X[:, 12] = np.random.normal(30, 13, n_samples)     # bun
    X[:, 13] = np.random.normal(0.95, 0.45, n_samples) # ast
    X[:, 14] = np.random.normal(0.75, 0.35, n_samples) # alt
    
    # Advanced medical markers
    X[:, 15] = np.random.normal(6.2, 2.5, n_samples)  # hba1c
    X[:, 16] = np.random.normal(135, 55, n_samples)    # gfr
    X[:, 17] = np.random.normal(0.46, 0.20, n_samples) # microalbumin
    X[:, 18] = np.random.normal(14.2, 4.5, n_samples)  # crp
    X[:, 19] = np.random.normal(83, 28, n_samples)     # vitamin_d
    X[:, 20] = np.random.normal(4.4, 2.1, n_samples)  # tsh
    X[:, 21] = np.random.normal(1.8, 0.7, n_samples)  # free_t4
    X[:, 22] = np.random.normal(155, 52, n_samples)    # ferritin
    X[:, 23] = np.random.normal(13.5, 5.0, n_samples)  # iron
    X[:, 24] = np.random.normal(285, 82, n_samples)    # b12
    
    # Cardiac markers
    X[:, 25] = np.random.normal(28, 16, n_samples)     # troponin
    X[:, 26] = np.random.normal(0.85, 0.5, n_samples) # bnp
    X[:, 27] = np.random.normal(4.5, 2.9, n_samples)  # d_dimer
    X[:, 28] = np.random.normal(94, 24, n_samples)     # fev1
    X[:, 29] = np.random.normal(84, 19, n_samples)     # fvc
    
    # Lifestyle factors
    X[:, 30] = np.random.exponential(18, n_samples)     # smoking_pack_years
    X[:, 31] = np.random.exponential(9, n_samples)      # alcohol_drinks_week
    X[:, 32] = np.random.normal(3.0, 2.5, n_samples)  # exercise_hours_week
    X[:, 33] = np.random.normal(6.5, 1.6, n_samples)  # sleep_hours
    X[:, 34] = np.random.exponential(3.5, n_samples)   # stress_level
    
    # Age groups
    elderly = X[:, 0] >= 65
    middle = (X[:, 0] >= 40) & (X[:, 0] < 65)
    young = X[:, 0] < 40
    
    # Age patterns
    X[elderly, 1] += np.random.normal(18, 10, elderly.sum())
    X[elderly, 6] += np.random.normal(30, 18, elderly.sum())
    X[elderly, 11] += np.random.normal(0.4, 0.15, elderly.sum())
    X[elderly, 16] -= np.random.normal(25, 15, elderly.sum())
    X[elderly, 25] += np.random.exponential(6, elderly.sum())
    
    X[young, 1] -= np.random.normal(8, 6, young.sum())
    X[young, 7] -= np.random.normal(10, 8, young.sum())
    X[young, 30] += np.random.exponential(6, young.sum())
    
    # Medical disease risk
    disease_risk = np.zeros(n_samples)
    
    # Major risk factors
    disease_risk += (X[:, 0] > 65) * 0.4
    disease_risk += (X[:, 5] > 30) * 0.25
    disease_risk += (X[:, 1] > 140) * 0.3
    disease_risk += (X[:, 6] > 130) * 0.2
    disease_risk += (X[:, 7] > 100) * 0.25
    disease_risk += (X[:, 10] > 200) * 0.15
    disease_risk += (X[:, 11] > 1.2) * 0.2
    disease_risk += (X[:, 15] > 6.5) * 0.22
    disease_risk += (X[:, 18] > 25) * 0.18
    disease_risk += (X[:, 25] > 40) * 0.35
    disease_risk += (X[:, 26] > 100) * 0.3
    disease_risk += (X[:, 30] > 25) * 0.15
    
    # Metabolic syndrome
    metabolic_score = (
        (X[:, 5] > 30) +
        (X[:, 1] > 130) +
        (X[:, 7] > 100) +
        (X[:, 8] < 1.0) +
        (X[:, 10] > 150)
    )
    disease_risk += (metabolic_score >= 3) * 0.28
    
    # Medical interactions
    disease_risk += ((X[:, 0] > 60) & (X[:, 1] > 135)) * 0.18
    disease_risk += ((X[:, 5] > 28) & (X[:, 7] > 90)) * 0.15
    disease_risk += ((X[:, 25] > 35) & (X[:, 26] > 80)) * 0.25
    disease_risk += ((X[:, 11] > 1.5) & (X[:, 16] < 50)) * 0.2
    
    # Add noise
    disease_risk += np.random.normal(0, 0.1, n_samples)
    
    # Calculate probability
    disease_prob = 1 / (1 + np.exp(-disease_risk))
    
    # Age-adjusted prevalence
    base_rate = 0.18
    elderly_boost = 0.1
    middle_boost = 0.05
    young_reduction = 0.04
    
    final_rate = np.full(n_samples, base_rate)
    final_rate[elderly] += elderly_boost
    final_rate[middle] += middle_boost
    final_rate[young] -= young_reduction
    
    disease_prob = disease_prob * final_rate / np.mean(disease_prob)
    disease_prob = np.clip(disease_prob, 0, 1)
    
    # Generate labels
    y = (np.random.random(n_samples) < disease_prob).astype(int)
    
    print(f"  ✅ Generated {n_samples:,} samples with {X.shape[1]} features")
    print(f"  📊 Disease prevalence: {np.mean(y)*100:.2f}%")
    
    return X, y

def create_fast_medical_features(X):
    """Create fast medical features"""
    print("🔧 Creating FAST Medical Features...")
    
    features = [X]
    
    # Medical ratios
    ratios = []
    ratios.append((X[:, 1] / X[:, 2]).reshape(-1, 1))  # BP ratio
    ratios.append((X[:, 6] / X[:, 8]).reshape(-1, 1))  # Total/HDL
    ratios.append((X[:, 9] / X[:, 8]).reshape(-1, 1))  # LDL/HDL
    ratios.append((X[:, 12] / X[:, 11]).reshape(-1, 1))  # BUN/Creatinine
    ratios.append((X[:, 16] / X[:, 0]).reshape(-1, 1))  # GFR/Age
    ratios.append((X[:, 25] / X[:, 26]).reshape(-1, 1))  # Troponin/BNP
    ratios.append((X[:, 28] / X[:, 29]).reshape(-1, 1))  # FEV1/FVC
    ratios.append((X[:, 0] * X[:, 5] / 1000).reshape(-1, 1))  # Age*BMI/1000
    
    if ratios:
        ratio_features = np.hstack(ratios)
        features.append(ratio_features)
    
    # Key polynomial features
    poly_features = np.hstack([
        X[:, 0:6] ** 2,  # Core vitals squared
        X[:, 6:11] ** 2,  # Blood tests squared
        X[:, 15:20] ** 2,  # Advanced markers squared
        (X[:, 0] * X[:, 1] / 100).reshape(-1, 1),  # Age*BP/100
        (X[:, 5] * X[:, 7] / 100).reshape(-1, 1),  # BMI*Glucose/100
        (X[:, 6] * X[:, 8] / 100).reshape(-1, 1),  # Chol*HDL/100
        (X[:, 11] * X[:, 16] / 100).reshape(-1, 1),  # Creatinine*GFR/100
        (X[:, 25] * X[:, 26] / 1000).reshape(-1, 1),  # Troponin*BNP/1000
        (X[:, 30] * X[:, 31] / 100).reshape(-1, 1),  # Smoking*Alcohol/100
    ])
    features.append(poly_features)
    
    X_fast = np.hstack(features)
    
    print(f"  ✅ Enhanced from {X.shape[1]} to {X_fast.shape[1]} features")
    return X_fast

def main():
    print("🏥 HEALTHCARE FAST MEDICAL AI")
    print("=" * 45)
    print("Fast medical AI approach to avoid hanging")
    
    # Generate data
    X, y = generate_fast_medical_data()
    
    # Create features
    X_fast = create_fast_medical_features(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_fast, y, test_size=0.2, random_state=888, stratify=y
    )
    
    # Feature selection
    selector = SelectKBest(f_classif, k=80)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Train model
    print("🤖 Training FAST Medical Model...")
    
    import time
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=888,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    training_time = time.time() - start_time
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎉 HEALTHCARE FAST MEDICAL RESULTS:")
    print(f"  📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  🎯 Previous: 84.99% → Current: {accuracy*100:.2f}%")
    print(f"  📈 Improvement: {accuracy*100 - 84.99:.2f}%")
    print(f"  ⏱️  Training Time: {training_time:.2f}s")
    print(f"  🧠 Features Used: {X_train_selected.shape[1]}")
    
    if accuracy >= 0.99:
        print(f"  🎉🎉🎉 FAST MEDICAL BREAKTHROUGH! 99%+ ACHIEVED! 🎉🎉🎉")
        status = "99%+ FAST MEDICAL BREAKTHROUGH"
    elif accuracy >= 0.98:
        print(f"  🚀 FAST MEDICAL EXCELLENCE! 98%+ ACHIEVED!")
        status = "98%+ FAST MEDICAL EXCELLENCE"
    elif accuracy >= 0.95:
        print(f"  ✅ FAST MEDICAL SUCCESS! 95%+ ACHIEVED!")
        status = "95%+ FAST MEDICAL SUCCESS"
    elif accuracy >= 0.90:
        print(f"  ✅ FAST MEDICAL IMPROVEMENT! 90%+ ACHIEVED!")
        status = "90%+ FAST MEDICAL IMPROVEMENT"
    else:
        print(f"  💡 FAST MEDICAL BASELINE: {accuracy*100:.1f}%")
        status = f"{accuracy*100:.1f}% FAST MEDICAL BASELINE"
    
    print(f"\n💎 FINAL STATUS: {status}")
    print(f"🔧 Fast Medical Techniques: 30K samples + Medical features + Fast RF")
    
    return accuracy

if __name__ == "__main__":
    accuracy = main()
    print(f"\n🎯 Healthcare Fast Medical Complete! Final Accuracy: {accuracy*100:.2f}%")
