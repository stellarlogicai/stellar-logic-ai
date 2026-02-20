#!/usr/bin/env python3
"""
FAST NEUROMORPHIC OPTIMIZER
Lightweight version to achieve 90%+ quickly (Current: 88.62%)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

class FastNeuromorphicOptimizer:
    def __init__(self):
        self.target_accuracy = 0.90
        self.current_accuracy = 0.8862
        
    def generate_neuromorphic_data(self, n_samples=50000):
        """Generate focused neuromorphic data"""
        print(f"🧠 Generating {n_samples:,} neuromorphic patterns...")
        
        # Core spiking features
        membrane_potential = np.random.normal(-65, 10, n_samples)
        threshold_potential = np.random.normal(-50, 5, n_samples)
        spike_rate = np.random.exponential(50, n_samples)
        
        # Synaptic features
        synaptic_weight = np.random.lognormal(0, 1, n_samples)
        ltp_strength = np.random.beta(3, 2, n_samples)
        stdp_window = np.random.exponential(20, n_samples)
        
        # Network features
        connectivity = np.random.beta(2, 2, n_samples)
        network_depth = np.random.randint(3, 10, n_samples)
        plasticity_rate = np.random.exponential(0.1, n_samples)
        
        # Performance features
        energy_efficiency = np.random.beta(4, 1, n_samples)
        inference_speed = np.random.exponential(1000, n_samples)
        noise_tolerance = np.random.beta(3, 1, n_samples)
        
        # Success criteria
        success_potential = (threshold_potential - membrane_potential > 10)
        success_synaptic = (synaptic_weight > 0.5) & (ltp_strength > 0.4)
        success_network = (connectivity > 0.3) & (network_depth > 4)
        success_performance = (energy_efficiency > 0.6) & (noise_tolerance > 0.5)
        
        # Combined success probability
        base_success = success_potential & success_synaptic & success_network & success_performance
        
        # Add realistic variation
        success_prob = base_success * (0.7 + 0.25 * np.random.random(n_samples))
        success_prob = np.clip(success_prob, 0.3, 0.9)
        
        y = (np.random.random(n_samples) < success_prob).astype(int)
        
        X = pd.DataFrame({
            'membrane_potential': membrane_potential,
            'threshold_potential': threshold_potential,
            'spike_rate': spike_rate,
            'synaptic_weight': synaptic_weight,
            'ltp_strength': ltp_strength,
            'stdp_window': stdp_window,
            'connectivity': connectivity,
            'network_depth': network_depth,
            'plasticity_rate': plasticity_rate,
            'energy_efficiency': energy_efficiency,
            'inference_speed': inference_speed,
            'noise_tolerance': noise_tolerance
        })
        
        return X, y
    
    def create_features(self, X):
        """Create key neuromorphic features"""
        X_feat = X.copy()
        
        # Ratios and interactions
        X_feat['threshold_gap'] = X['threshold_potential'] - X['membrane_potential']
        X_feat['spike_efficiency'] = X['spike_rate'] * X['synaptic_weight']
        X_feat['network_complexity'] = X['connectivity'] * X['network_depth']
        X_feat['plasticity_strength'] = X['ltp_strength'] * X['plasticity_rate']
        X_feat['performance_score'] = X['energy_efficiency'] * X['noise_tolerance']
        
        # Log transforms
        X_feat['spike_rate_log'] = np.log1p(X['spike_rate'])
        X_feat['inference_speed_log'] = np.log1p(X['inference_speed'])
        
        return X_feat
    
    def create_ensemble(self):
        """Create optimized ensemble"""
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=123
        )
        
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=456,
            early_stopping=True
        )
        
        return VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('mlp', mlp)],
            voting='soft',
            weights=[2, 2, 1]
        )
    
    def optimize(self):
        """Main optimization"""
        print("\n🧠 FAST NEUROMORPHIC OPTIMIZER")
        print("=" * 50)
        print(f"Target: {self.target_accuracy*100:.1f}% | Current: {self.current_accuracy*100:.2f}%")
        print("=" * 50)
        
        start_time = time.time()
        
        # Generate data
        X, y = self.generate_neuromorphic_data(50000)
        
        # Create features
        X_enhanced = self.create_features(X)
        
        # Feature selection
        selector = SelectKBest(f_classif, k=15)
        X_selected = selector.fit_transform(X_enhanced, y)
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train
        print("🎯 Training ensemble...")
        ensemble = self.create_ensemble()
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        train_acc = ensemble.score(X_train, y_train)
        
        elapsed = time.time() - start_time
        improvement = accuracy - self.current_accuracy
        
        print(f"\n🎉 RESULTS:")
        print(f"   🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   📊 Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   ⏱️  Time: {elapsed:.1f}s")
        print(f"   📈 Improvement: +{improvement*100:.2f}%")
        
        if accuracy >= self.target_accuracy:
            print(f"   ✅ SUCCESS: Achieved 90%+ target!")
        else:
            gap = self.target_accuracy - accuracy
            print(f"   ⚠️  Gap: {gap*100:.2f}%")
        
        return {
            'test_accuracy': accuracy,
            'train_accuracy': train_acc,
            'improvement': improvement,
            'time': elapsed
        }

if __name__ == "__main__":
    from sklearn.ensemble import VotingClassifier
    optimizer = FastNeuromorphicOptimizer()
    results = optimizer.optimize()
