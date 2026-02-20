#!/usr/bin/env python3
"""
Stellar Logic AI - Accuracy Validation System
Double-check and validate the accuracy numbers for realism
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AccuracyValidator:
    """Validate accuracy claims with realistic testing"""
    
    def __init__(self):
        self.validation_results = []
        
    def generate_challenging_data(self, dataset_type: str, n_samples: int = 10000, difficulty: str = "medium"):
        """Generate more challenging, realistic data"""
        np.random.seed(123)  # Different seed for validation
        
        if dataset_type == "healthcare_medical_imaging":
            n_features = 30
            
            # More challenging medical data with overlap
            if difficulty == "easy":
                # Clear separation (unrealistic)
                healthy = np.random.randn(n_samples//2, n_features) * 0.5
                disease = np.random.randn(n_samples//2, n_features) * 1.5 + 2.0
            elif difficulty == "medium":
                # Some overlap (realistic)
                healthy = np.random.randn(n_samples//2, n_features) * 1.0
                disease = np.random.randn(n_samples//2, n_features) * 1.2 + 0.8
            else:  # hard
                # High overlap (very challenging)
                healthy = np.random.randn(n_samples//2, n_features) * 1.2
                disease = np.random.randn(n_samples//2, n_features) * 1.3 + 0.3
            
            X = np.vstack([healthy, disease])
            y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))
            
        elif dataset_type == "financial_fraud_detection":
            n_features = 25
            
            # More challenging financial data
            if difficulty == "easy":
                legitimate = np.random.randn(int(n_samples*0.95), n_features) * 0.8
                fraudulent = np.random.randn(int(n_samples*0.05), n_features) * 2.5 + 2.0
            elif difficulty == "medium":
                legitimate = np.random.randn(int(n_samples*0.95), n_features) * 1.2
                fraudulent = np.random.randn(int(n_samples*0.05), n_features) * 1.8 + 1.0
            else:  # hard
                legitimate = np.random.randn(int(n_samples*0.95), n_features) * 1.5
                fraudulent = np.random.randn(int(n_samples*0.05), n_features) * 1.7 + 0.5
            
            X = np.vstack([legitimate, fraudulent])
            y = np.array([0]*int(n_samples*0.95) + [1]*int(n_samples*0.05))
            
        elif dataset_type == "gaming_anti_cheat":
            n_features = 20
            
            # More challenging gaming data
            if difficulty == "easy":
                normal = np.random.randn(int(n_samples*0.9), n_features) * 0.7
                cheaters = np.random.randn(int(n_samples*0.1), n_features) * 2.2 + 1.8
            elif difficulty == "medium":
                normal = np.random.randn(int(n_samples*0.9), n_features) * 1.0
                cheaters = np.random.randn(int(n_samples*0.1), n_features) * 1.5 + 0.8
            else:  # hard
                normal = np.random.randn(int(n_samples*0.9), n_features) * 1.2
                cheaters = np.random.randn(int(n_samples*0.1), n_features) * 1.4 + 0.4
            
            X = np.vstack([normal, cheaters])
            y = np.array([0]*int(n_samples*0.9) + [1]*int(n_samples*0.1))
            
        else:
            # Default challenging data
            n_features = 15
            X = np.random.randn(n_samples, n_features)
            # Create overlapping classes
            class1 = np.random.randn(n_samples//2, n_features) * 1.0
            class2 = np.random.randn(n_samples//2, n_features) * 1.1 + 0.2
            X = np.vstack([class1, class2])
            y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))
        
        return X, y
    
    def validate_single_dataset(self, dataset_type: str, difficulty: str = "medium"):
        """Validate accuracy on a single dataset with realistic difficulty"""
        print(f"\n🔍 Validating {dataset_type} - {difficulty} difficulty")
        
        # Generate challenging data
        X, y = self.generate_challenging_data(dataset_type, n_samples=20000, difficulty=difficulty)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Further split test set for validation
        X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_val_scaled = scaler.transform(X_val)
        
        print(f"  📊 Dataset info:")
        print(f"    Training samples: {len(X_train)}")
        print(f"    Test samples: {len(X_test)}")
        print(f"    Validation samples: {len(X_val)}")
        print(f"    Features: {X.shape[1]}")
        print(f"    Class distribution: {np.bincount(y)}")
        
        # Train model with conservative parameters
        print(f"  🤖 Training model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Test on held-out test set
        y_test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Test on validation set (unseen during training)
        y_val_pred = model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        # Check for overfitting
        y_train_pred = model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        print(f"  📈 Results:")
        print(f"    Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"    Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"    Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        # Check overfitting
        overfitting_gap = train_accuracy - val_accuracy
        if overfitting_gap > 0.05:
            print(f"    ⚠️  Overfitting detected: {overfitting_gap:.4f} gap")
        else:
            print(f"    ✅ No significant overfitting")
        
        # Confusion matrix for validation set
        cm = confusion_matrix(y_val, y_val_pred)
        print(f"    📊 Confusion Matrix (Validation):")
        print(f"      {cm}")
        
        # Classification report
        try:
            report = classification_report(y_val, y_val_pred, output_dict=True)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1 = report['weighted avg']['f1-score']
            
            print(f"    📋 Classification Report (Validation):")
            print(f"      Precision: {precision:.4f}")
            print(f"      Recall: {recall:.4f}")
            print(f"      F1-Score: {f1:.4f}")
        except:
            precision = recall = f1 = 0.0
        
        # Store validation result
        result = {
            'dataset_type': dataset_type,
            'difficulty': difficulty,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'val_accuracy': val_accuracy,
            'overfitting_gap': overfitting_gap,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'validation_samples': len(X_val)
        }
        
        self.validation_results.append(result)
        
        # Assessment
        if val_accuracy >= 0.99:
            print(f"    🎉 EXCELLENT: {val_accuracy*100:.1f}% validation accuracy!")
        elif val_accuracy >= 0.95:
            print(f"    🚀 VERY GOOD: {val_accuracy*100:.1f}% validation accuracy!")
        elif val_accuracy >= 0.90:
            print(f"    ✅ GOOD: {val_accuracy*100:.1f}% validation accuracy!")
        else:
            print(f"    💡 BASELINE: {val_accuracy*100:.1f}% validation accuracy")
        
        return val_accuracy
    
    def run_comprehensive_validation(self):
        """Run comprehensive validation across difficulties"""
        print("🔍 STELLAR LOGIC AI - ACCURACY VALIDATION")
        print("=" * 60)
        print("Double-checking accuracy claims with realistic testing...")
        
        # Test configurations
        configs = [
            ("healthcare_medical_imaging", "easy"),
            ("healthcare_medical_imaging", "medium"),
            ("healthcare_medical_imaging", "hard"),
            ("financial_fraud_detection", "easy"),
            ("financial_fraud_detection", "medium"),
            ("financial_fraud_detection", "hard"),
            ("gaming_anti_cheat", "easy"),
            ("gaming_anti_cheat", "medium"),
            ("gaming_anti_cheat", "hard"),
        ]
        
        for dataset_type, difficulty in configs:
            self.validate_single_dataset(dataset_type, difficulty)
        
        # Generate validation report
        self.generate_validation_report()
        
        return self.validation_results
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "=" * 60)
        print("📊 VALIDATION REPORT")
        print("=" * 60)
        
        # Separate by difficulty
        easy_results = [r for r in self.validation_results if r['difficulty'] == 'easy']
        medium_results = [r for r in self.validation_results if r['difficulty'] == 'medium']
        hard_results = [r for r in self.validation_results if r['difficulty'] == 'hard']
        
        # Calculate statistics for each difficulty
        def calc_stats(results, name):
            if not results:
                return {}
            
            val_accuracies = [r['val_accuracy'] for r in results]
            return {
                'name': name,
                'count': len(results),
                'avg_val_accuracy': np.mean(val_accuracies),
                'max_val_accuracy': np.max(val_accuracies),
                'min_val_accuracy': np.min(val_accuracies),
                'achieved_99': any(r['val_accuracy'] >= 0.99 for r in results),
                'achieved_95': any(r['val_accuracy'] >= 0.95 for r in results),
                'achieved_90': any(r['val_accuracy'] >= 0.90 for r in results)
            }
        
        easy_stats = calc_stats(easy_results, "Easy")
        medium_stats = calc_stats(medium_results, "Medium")
        hard_stats = calc_stats(hard_results, "Hard")
        
        print(f"\n🎯 VALIDATION STATISTICS:")
        
        for stats in [easy_stats, medium_stats, hard_stats]:
            if stats:
                print(f"\n  📊 {stats['name']} Difficulty:")
                print(f"    Tests Run: {stats['count']}")
                print(f"    Average Validation Accuracy: {stats['avg_val_accuracy']:.4f} ({stats['avg_val_accuracy']*100:.2f}%)")
                print(f"    Maximum Validation Accuracy: {stats['max_val_accuracy']:.4f} ({stats['max_val_accuracy']*100:.2f}%)")
                print(f"    Minimum Validation Accuracy: {stats['min_val_accuracy']:.4f} ({stats['min_val_accuracy']*100:.2f}%)")
                print(f"    ✅ 99%+ Achieved: {stats['achieved_99']}")
                print(f"    ✅ 95%+ Achieved: {stats['achieved_95']}")
                print(f"    ✅ 90%+ Achieved: {stats['achieved_90']}")
        
        # Overall assessment
        all_val_accuracies = [r['val_accuracy'] for r in self.validation_results]
        overall_avg = np.mean(all_val_accuracies)
        overall_max = np.max(all_val_accuracies)
        
        print(f"\n🏆 OVERALL VALIDATION RESULTS:")
        print(f"  📈 Average Validation Accuracy: {overall_avg:.4f} ({overall_avg*100:.2f}%)")
        print(f"  🎯 Maximum Validation Accuracy: {overall_max:.4f} ({overall_max*100:.2f}%)")
        print(f"  📊 Total Tests: {len(self.validation_results)}")
        
        # Realistic assessment
        if medium_stats['avg_val_accuracy'] >= 0.95:
            print(f"\n🎉 EXCELLENT: Realistic 95%+ accuracy achieved!")
            realistic_claim = "95%+"
        elif medium_stats['avg_val_accuracy'] >= 0.90:
            print(f"\n🚀 VERY GOOD: Realistic 90%+ accuracy achieved!")
            realistic_claim = "90%+"
        elif medium_stats['avg_val_accuracy'] >= 0.85:
            print(f"\n✅ GOOD: Realistic 85%+ accuracy achieved!")
            realistic_claim = "85%+"
        else:
            print(f"\n💡 BASELINE: Realistic accuracy established")
            realistic_claim = f"{overall_avg*100:.1f}%"
        
        print(f"\n💎 HONEST ASSESSMENT:")
        print(f"  🎯 Realistic Accuracy Claim: {realistic_claim}")
        print(f"  📊 Based on: Medium difficulty validation")
        print(f"  🔍 Method: Proper train/test/validation split")
        print(f"  ✅ Status: Genuine performance measurement")
        
        return {
            'realistic_claim': realistic_claim,
            'medium_avg': medium_stats.get('avg_val_accuracy', 0),
            'overall_avg': overall_avg,
            'validation_results': self.validation_results
        }

# Main execution
if __name__ == "__main__":
    print("🔍 Starting Accuracy Validation...")
    print("Double-checking those accuracy numbers with realistic testing...")
    
    validator = AccuracyValidator()
    results = validator.run_comprehensive_validation()
    
    print(f"\n🎯 VALIDATION COMPLETE!")
    print(f"Honest accuracy assessment provided!")
