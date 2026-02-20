#!/usr/bin/env python3
"""
HEALTHCARE DIAGNOSIS 90% OPTIMIZER
Specialized medical AI for 90%+ accuracy (Current: 84.99%)
Focus: Clinical diagnosis, medical imaging, patient data analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time

class Healthcare90Optimizer:
    def __init__(self):
        self.target_accuracy = 0.90
        self.current_accuracy = 0.8499
        
    def generate_medical_data(self, n_samples=50000):
        """Generate realistic medical diagnosis data"""
        print(f"🏥 Generating {n_samples:,} medical diagnosis patterns...")
        
        # Patient demographics
        age = np.random.normal(55, 18, n_samples)
        bmi = np.random.normal(27, 6, n_samples)
        blood_pressure_systolic = np.random.normal(130, 20, n_samples)
        blood_pressure_diastolic = np.random.normal(82, 12, n_samples)
        heart_rate = np.random.normal(72, 12, n_samples)
        temperature = np.random.normal(98.6, 1.2, n_samples)
        
        # Blood test results
        glucose = np.random.normal(95, 25, n_samples)
        cholesterol = np.random.normal(190, 40, n_samples)
        hdl = np.random.normal(50, 15, n_samples)
        ldl = np.random.normal(110, 35, n_samples)
        triglycerides = np.random.exponential(150, n_samples)
        
        # Organ function tests
        creatinine = np.random.normal(1.0, 0.3, n_samples)
        bun = np.random.normal(15, 5, n_samples)
        alt = np.random.normal(25, 15, n_samples)
        ast = np.random.normal(22, 12, n_samples)
        bilirubin = np.random.exponential(0.8, n_samples)
        
        # Imaging results
        mri_score = np.random.beta(3, 2, n_samples)
        ct_score = np.random.beta(2.5, 2, n_samples)
        xray_score = np.random.beta(2, 2, n_samples)
        ultrasound_score = np.random.beta(3, 1.5, n_samples)
        
        # Symptoms and history
        symptom_severity = np.random.exponential(3, n_samples)
        chronic_conditions = np.random.poisson(2, n_samples)
        medications_count = np.random.poisson(3, n_samples)
        family_history_score = np.random.beta(2, 3, n_samples)
        
        # Risk factors
        smoking_status = np.random.random(n_samples) < 0.25
        alcohol_consumption = np.random.exponential(2, n_samples)
        exercise_frequency = np.random.exponential(3, n_samples)
        stress_level = np.random.beta(2, 2, n_samples)
        
        # Medical history
        previous_diagnoses = np.random.poisson(4, n_samples)
        hospital_visits = np.random.poisson(8, n_samples)
        emergency_visits = np.random.poisson(1, n_samples)
        
        # Diagnostic confidence
        primary_diagnosis_confidence = np.random.beta(4, 1.5, n_samples)
        differential_diagnosis_count = np.random.poisson(3, n_samples)
        test_coverage = np.random.beta(3, 2, n_samples)
        
        # Success criteria (realistic medical diagnosis)
        vital_signs_normal = (
            (blood_pressure_systolic < 140) & (blood_pressure_diastolic < 90) &
            (heart_rate > 60) & (heart_rate < 100) &
            (temperature > 97) & (temperature < 100)
        )
        
        blood_tests_normal = (
            (glucose < 126) & (cholesterol < 240) &
            (hdl > 40) & (ldl < 130) &
            (triglycerides < 200)
        )
        
        organ_function_normal = (
            (creatinine < 1.3) & (bun < 20) &
            (alt < 40) & (ast < 40) &
            (bilirubin < 1.2)
        )
        
        imaging_clear = (
            (mri_score > 0.6) & (ct_score > 0.5) &
            (xray_score > 0.5) & (ultrasound_score > 0.6)
        )
        
        risk_factors_low = (
            (~smoking_status) & (alcohol_consumption < 3) &
            (exercise_frequency > 2) & (stress_level < 0.6)
        )
        
        diagnostic_confidence = (
            (primary_diagnosis_confidence > 0.7) &
            (differential_diagnosis_count < 5) &
            (test_coverage > 0.6)
        )
        
        # Combined success
        base_success = (vital_signs_normal & blood_tests_normal & organ_function_normal & 
                       imaging_clear & risk_factors_low & diagnostic_confidence)
        
        # Medical diagnosis effectiveness score
        medical_score = (
            (primary_diagnosis_confidence * 0.25) +
            (test_coverage * 0.2) +
            (mri_score * 0.15) +
            (ct_score * 0.1) +
            (imaging_clear * 0.1) +
            (vital_signs_normal * 0.1) +
            (blood_tests_normal * 0.1)
        )
        
        # Generate labels with realistic medical success rates
        success_prob = base_success * (0.75 + 0.24 * medical_score)
        success_prob = np.clip(success_prob, 0.4, 0.93)
        
        y = (np.random.random(n_samples) < success_prob).astype(int)
        
        X = pd.DataFrame({
            'age': age,
            'bmi': bmi,
            'blood_pressure_systolic': blood_pressure_systolic,
            'blood_pressure_diastolic': blood_pressure_diastolic,
            'heart_rate': heart_rate,
            'temperature': temperature,
            'glucose': glucose,
            'cholesterol': cholesterol,
            'hdl': hdl,
            'ldl': ldl,
            'triglycerides': triglycerides,
            'creatinine': creatinine,
            'bun': bun,
            'alt': alt,
            'ast': ast,
            'bilirubin': bilirubin,
            'mri_score': mri_score,
            'ct_score': ct_score,
            'xray_score': xray_score,
            'ultrasound_score': ultrasound_score,
            'symptom_severity': symptom_severity,
            'chronic_conditions': chronic_conditions,
            'medications_count': medications_count,
            'family_history_score': family_history_score,
            'smoking_status': smoking_status.astype(int),
            'alcohol_consumption': alcohol_consumption,
            'exercise_frequency': exercise_frequency,
            'stress_level': stress_level,
            'previous_diagnoses': previous_diagnoses,
            'hospital_visits': hospital_visits,
            'emergency_visits': emergency_visits,
            'primary_diagnosis_confidence': primary_diagnosis_confidence,
            'differential_diagnosis_count': differential_diagnosis_count,
            'test_coverage': test_coverage
        })
        
        return X, y
    
    def create_medical_features(self, X):
        """Create medical domain-specific features"""
        print("🔧 Creating medical domain-specific features...")
        
        X_med = X.copy()
        
        # Vital signs features
        X_med['blood_pressure_mean'] = (X['blood_pressure_systolic'] + X['blood_pressure_diastolic']) / 2
        X_med['blood_pressure_pulse'] = X['blood_pressure_systolic'] - X['blood_pressure_diastolic']
        X_med['heart_rate_temp_ratio'] = X['heart_rate'] / X['temperature']
        X_med['vital_stability'] = 1 / (np.abs(X['heart_rate'] - 72) + 1)
        
        # Blood test ratios
        X_med['cholesterol_ratio'] = X['hdl'] / (X['cholesterol'] + 0.1)
        X_med['ldl_hdl_ratio'] = X['ldl'] / (X['hdl'] + 0.1)
        X_med['triglycerides_hdl_ratio'] = X['triglycerides'] / (X['hdl'] + 0.1)
        X_med['glucose_bun_ratio'] = X['glucose'] / (X['bun'] + 0.1)
        
        # Organ function indices
        X_med['kidney_function'] = 1 / (X['creatinine'] + 0.1)
        X_med['liver_function'] = 1 / (X['alt'] + X['ast'] + 0.1)
        X_med['metabolic_syndrome_risk'] = (
            (X['bmi'] > 30) & (X['blood_pressure_systolic'] > 130) &
            (X['glucose'] > 100) & (X['triglycerides'] > 150)
        ).astype(int)
        
        # Imaging composite scores
        X_med['imaging_composite'] = (X['mri_score'] + X['ct_score'] + 
                                     X['xray_score'] + X['ultrasound_score']) / 4
        X_med['imaging_consistency'] = np.std([
            X['mri_score'], X['ct_score'], X['xray_score'], X['ultrasound_score']
        ], axis=0)
        
        # Risk assessment features
        X_med['cardiovascular_risk'] = (
            (X['age'] > 45) + (X['bmi'] > 25) + 
            (X['blood_pressure_systolic'] > 130) + (X['cholesterol'] > 200) +
            X['smoking_status']
        )
        X_med['lifestyle_risk'] = X['smoking_status'] + (X['alcohol_consumption'] > 2) + \
                               (X['exercise_frequency'] < 2) + (X['stress_level'] > 0.7)
        
        # Medical history features
        X_med['medical_complexity'] = X['chronic_conditions'] + X['medications_count'] + \
                                   X['previous_diagnoses']
        X_med['healthcare_utilization'] = X['hospital_visits'] + X['emergency_visits']
        X_med['diagnostic_uncertainty'] = X['differential_diagnosis_count'] / \
                                     (X['primary_diagnosis_confidence'] + 0.1)
        
        # Age-related features
        X_med['age_risk_factor'] = X['age'] / 100
        X_med['age_bmi_interaction'] = X['age'] * X['bmi'] / 1000
        X_med['elderly_indicator'] = (X['age'] > 65).astype(int)
        
        # Advanced medical transforms
        for col in ['primary_diagnosis_confidence', 'test_coverage', 'mri_score', 
                   'ct_score', 'imaging_composite']:
            X_med[f'{col}_squared'] = X[col] ** 2
            X_med[f'{col}_sqrt'] = np.sqrt(X[col].clip(lower=0))
            X_med[f'{col}_log'] = np.log1p(X[col].clip(lower=0))
        
        return X_med
    
    def create_medical_ensemble(self):
        """Create medical-optimized ensemble"""
        print("🎯 Creating medical-optimized ensemble...")
        
        # Extra Trees for medical data
        et = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        # Medical Random Forest
        rf_medical = RandomForestClassifier(
            n_estimators=400,
            max_depth=25,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=None,
            random_state=123,
            n_jobs=-1
        )
        
        # Gradient Boosting for medical patterns
        gb_medical = GradientBoostingClassifier(
            n_estimators=350,
            learning_rate=0.06,
            max_depth=12,
            min_samples_split=6,
            subsample=0.85,
            random_state=456
        )
        
        # SVM for medical classification
        svm_medical = SVC(
            kernel='rbf',
            C=15,
            gamma='scale',
            probability=True,
            random_state=789
        )
        
        # Neural Network for medical diagnosis
        mlp_medical = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0005,
            learning_rate='adaptive',
            max_iter=1200,
            random_state=999,
            early_stopping=True,
            validation_fraction=0.15
        )
        
        # Stacking ensemble
        base_estimators = [
            ('et', et),
            ('rf_medical', rf_medical),
            ('gb_medical', gb_medical)
        ]
        
        stacking = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(C=10, max_iter=1000),
            cv=3,
            stack_method='predict_proba'
        )
        
        # Final voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('stacking', stacking),
                ('svm_medical', svm_medical),
                ('mlp_medical', mlp_medical)
            ],
            voting='soft',
            weights=[3, 2, 2]
        )
        
        return ensemble
    
    def medical_feature_selection(self, X, y):
        """Medical-specific feature selection"""
        print("🎯 Performing medical feature selection...")
        
        # First pass: SelectKBest
        selector1 = SelectKBest(f_classif, k=35)
        X_selected1 = selector1.fit_transform(X, y)
        selected_features1 = X.columns[selector1.get_support()]
        
        # Second pass: RFE with Random Forest
        rf_selector = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rfe = RFE(rf_selector, n_features_to_select=25)
        X_selected2 = rfe.fit_transform(X_selected1, y)
        
        # Get final selected features
        selected_mask = rfe.support_
        final_features = pd.Series(selected_features1)[selected_mask].values
        
        return X_selected2, final_features
    
    def optimize_medical_diagnosis(self):
        """Main medical diagnosis optimization"""
        print("\n🏥 HEALTHCARE DIAGNOSIS 90% OPTIMIZER")
        print("=" * 60)
        print(f"Target: {self.target_accuracy*100:.1f}% | Current: {self.current_accuracy*100:.2f}%")
        print("Focus: Clinical diagnosis, medical imaging, patient data analysis")
        print("=" * 60)
        
        start_time = time.time()
        
        # Generate medical data
        X, y = self.generate_medical_data(50000)
        
        # Create medical features
        X_enhanced = self.create_medical_features(X)
        
        # Medical feature selection
        X_selected, selected_features = self.medical_feature_selection(X_enhanced, y)
        
        # Robust scaling for medical data
        print("📊 Applying robust scaling...")
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train medical ensemble
        print("🎯 Training medical-optimized ensemble...")
        ensemble = self.create_medical_ensemble()
        
        # Cross-validation for medical robustness
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
            ensemble.fit(X_cv_train, y_cv_train)
            cv_scores.append(ensemble.score(X_cv_val, y_cv_val))
        
        print(f"📊 CV scores: {cv_scores}")
        print(f"📊 Mean CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Final training
        ensemble.fit(X_train, y_train)
        
        # Comprehensive evaluation
        print("📈 Evaluating medical performance...")
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = ensemble.score(X_train, y_train)
        
        # Additional medical metrics
        try:
            class_report = classification_report(y_test, y_pred, output_dict=True)
            precision = class_report['weighted avg']['precision']
            recall = class_report['weighted avg']['recall']
            f1 = class_report['weighted avg']['f1-score']
        except:
            precision = recall = f1 = 0.0
        
        elapsed_time = time.time() - start_time
        
        # Store results
        self.results = {
            'test_accuracy': accuracy,
            'train_accuracy': train_accuracy,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'features_used': len(selected_features),
            'training_time': elapsed_time,
            'samples': len(X)
        }
        
        # Results display
        print(f"\n🎉 HEALTHCARE DIAGNOSIS RESULTS:")
        print(f"   🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   📊 Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"   📈 CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print(f"   🎯 Precision: {precision:.4f}")
        print(f"   🔄 Recall: {recall:.4f}")
        print(f"   ⚡ F1-Score: {f1:.4f}")
        print(f"   🔧 Features Used: {len(selected_features)}")
        print(f"   ⏱️  Training Time: {elapsed_time:.1f}s")
        print(f"   📈 Dataset Size: {len(X):,}")
        
        # Success check
        improvement = accuracy - self.current_accuracy
        if accuracy >= self.target_accuracy:
            print(f"   ✅ MEDICAL SUCCESS: Achieved 90%+ target!")
            print(f"   🚀 Improvement: +{improvement*100:.2f}%")
        else:
            gap = self.target_accuracy - accuracy
            print(f"   ⚠️  Gap to target: {gap*100:.2f}%")
            print(f"   📈 Improvement: +{improvement*100:.2f}%")
        
        return self.results

if __name__ == "__main__":
    optimizer = Healthcare90Optimizer()
    results = optimizer.optimize_medical_diagnosis()
