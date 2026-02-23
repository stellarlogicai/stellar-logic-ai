#!/usr/bin/env python3
"""
ADVANCED RISK SCORING ENGINE
Gradient boosting ensemble with dynamic learning and real-time adaptation
"""

import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

class AdvancedRiskScoringEngine:
    """Advanced risk scoring with ensemble methods and dynamic learning"""
    
    def __init__(self):
        self.base_path = "c:/Users/merce/Documents/helm-ai"
        self.production_path = "c:/Users/merce/Documents/helm-ai/production"
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.production_path, "logs/advanced_risk_scoring.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Training data storage
        self.training_data = []
        self.prediction_history = deque(maxlen=10000)
        self.feedback_data = deque(maxlen=5000)
        
        # Risk thresholds (dynamic)
        self.risk_thresholds = {
            'minimal': 0.2,
            'low': 0.4,
            'medium': 0.6,
            'high': 0.8,
            'critical': 1.0
        }
        
        # Performance metrics
        self.metrics = {
            'total_predictions': 0,
            'high_risk_predictions': 0,
            'model_retrains': 0,
            'accuracy_improvements': 0,
            'last_retrain': None,
            'avg_prediction_time_ms': 0
        }
        
        # Feature definitions
        self.feature_columns = [
            'accuracy', 'kills_per_game', 'deaths_per_game', 'win_rate',
            'avg_session_length', 'login_frequency', 'playtime_variance',
            'performance_consistency', 'connected_accounts_count', 'account_age_days',
            'peak_hours_count', 'weekend_ratio', 'recent_anomaly_count',
            'cross_account_similarity', 'device_fingerprint_score', 'ip_risk_score'
        ]
        
        # Load existing models
        self.load_existing_models()
        
        self.logger.info("Advanced Risk Scoring Engine initialized")
    
    def load_existing_models(self):
        """Load pre-trained models if available"""
        models_dir = os.path.join(self.production_path, "risk_models")
        
        if os.path.exists(models_dir):
            try:
                # Load Gradient Boosting
                gb_path = os.path.join(models_dir, "gradient_boosting.pkl")
                if os.path.exists(gb_path):
                    self.models['gradient_boosting'] = joblib.load(gb_path)
                    self.logger.info("Loaded Gradient Boosting model")
                
                # Load XGBoost
                xgb_path = os.path.join(models_dir, "xgboost.pkl")
                if os.path.exists(xgb_path):
                    self.models['xgboost'] = joblib.load(xgb_path)
                    self.logger.info("Loaded XGBoost model")
                
                # Load LightGBM
                lgb_path = os.path.join(models_dir, "lightgbm.pkl")
                if os.path.exists(lgb_path):
                    self.models['lightgbm'] = joblib.load(lgb_path)
                    self.logger.info("Loaded LightGBM model")
                
                # Load Random Forest
                rf_path = os.path.join(models_dir, "random_forest.pkl")
                if os.path.exists(rf_path):
                    self.models['random_forest'] = joblib.load(rf_path)
                    self.logger.info("Loaded Random Forest model")
                
                # Load scalers
                scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
                if os.path.exists(scaler_path):
                    self.scalers['feature'] = joblib.load(scaler_path)
                    self.logger.info("Loaded feature scaler")
                
                # Load thresholds
                thresholds_path = os.path.join(models_dir, "risk_thresholds.json")
                if os.path.exists(thresholds_path):
                    with open(thresholds_path, 'r') as f:
                        self.risk_thresholds = json.load(f)
                    self.logger.info("Loaded dynamic risk thresholds")
                
            except Exception as e:
                self.logger.error(f"Error loading existing models: {str(e)}")
        else:
            self.logger.info("No existing models found - will train new ones")
    
    def create_ensemble_models(self):
        """Create ensemble of advanced ML models"""
        models = {}
        
        # Gradient Boosting Classifier
        models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        
        # XGBoost
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        # LightGBM
        models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        # Random Forest (for diversity)
        models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        return models
    
    def generate_training_data(self, n_samples=5000):
        """Generate realistic training data for risk scoring"""
        self.logger.info(f"Generating {n_samples} training samples...")
        
        np.random.seed(42)
        data = []
        
        for i in range(n_samples):
            # Generate legitimate user profiles (70%)
            if i < n_samples * 0.7:
                sample = {
                    'accuracy': np.random.normal(45, 15),
                    'kills_per_game': np.random.normal(8, 3),
                    'deaths_per_game': np.random.normal(10, 4),
                    'win_rate': np.random.beta(2, 3),  # Beta distribution for win rates
                    'avg_session_length': np.random.normal(90, 30),
                    'login_frequency': np.random.poisson(2),
                    'playtime_variance': np.random.exponential(1000),
                    'performance_consistency': np.random.beta(3, 2),
                    'connected_accounts_count': np.random.poisson(1),
                    'account_age_days': np.random.normal(200, 100),
                    'peak_hours_count': np.random.randint(1, 4),
                    'weekend_ratio': np.random.beta(2, 3),
                    'recent_anomaly_count': np.random.poisson(0.5),
                    'cross_account_similarity': np.random.beta(2, 3),
                    'device_fingerprint_score': np.random.normal(0.3, 0.1),
                    'ip_risk_score': np.random.exponential(0.1),
                    'is_high_risk': 0
                }
            else:
                # Generate high-risk user profiles (30%)
                sample = {
                    'accuracy': np.random.normal(85, 10),  # Higher accuracy
                    'kills_per_game': np.random.normal(20, 5),  # More kills
                    'deaths_per_game': np.random.normal(3, 2),  # Fewer deaths
                    'win_rate': np.random.beta(5, 1),  # Higher win rate
                    'avg_session_length': np.random.normal(180, 60),  # Longer sessions
                    'login_frequency': np.random.poisson(8),  # More frequent logins
                    'playtime_variance': np.random.exponential(5000),  # Higher variance
                    'performance_consistency': np.random.beta(5, 1),  # Too consistent
                    'connected_accounts_count': np.random.poisson(5),  # More connected accounts
                    'account_age_days': np.random.normal(30, 20),  # Newer accounts
                    'peak_hours_count': np.random.randint(3, 8),
                    'weekend_ratio': np.random.beta(1, 1),
                    'recent_anomaly_count': np.random.poisson(3),
                    'cross_account_similarity': np.random.beta(4, 1),  # High similarity
                    'device_fingerprint_score': np.random.normal(0.8, 0.1),  # Suspicious device
                    'ip_risk_score': np.random.exponential(0.5),  # Higher IP risk
                    'is_high_risk': 1
                }
            
            # Ensure values are in reasonable ranges
            sample['accuracy'] = np.clip(sample['accuracy'], 0, 100)
            sample['kills_per_game'] = max(0, sample['kills_per_game'])
            sample['deaths_per_game'] = max(0, sample['deaths_per_game'])
            sample['win_rate'] = np.clip(sample['win_rate'], 0, 1)
            sample['avg_session_length'] = max(0, sample['avg_session_length'])
            sample['login_frequency'] = max(0, sample['login_frequency'])
            sample['performance_consistency'] = np.clip(sample['performance_consistency'], 0, 1)
            sample['connected_accounts_count'] = max(0, sample['connected_accounts_count'])
            sample['account_age_days'] = max(1, sample['account_age_days'])
            sample['weekend_ratio'] = np.clip(sample['weekend_ratio'], 0, 1)
            sample['recent_anomaly_count'] = max(0, sample['recent_anomaly_count'])
            sample['cross_account_similarity'] = np.clip(sample['cross_account_similarity'], 0, 1)
            sample['device_fingerprint_score'] = np.clip(sample['device_fingerprint_score'], 0, 1)
            sample['ip_risk_score'] = np.clip(sample['ip_risk_score'], 0, 1)
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def train_ensemble_models(self, training_data=None):
        """Train ensemble of risk scoring models"""
        self.logger.info("Training ensemble risk scoring models...")
        
        # Generate training data if not provided
        if training_data is None:
            training_data = self.generate_training_data()
        
        # Prepare features and target
        X = training_data[self.feature_columns]
        y = training_data['is_high_risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        if 'feature' not in self.scalers:
            self.scalers['feature'] = StandardScaler()
        
        X_train_scaled = self.scalers['feature'].fit_transform(X_train)
        X_test_scaled = self.scalers['feature'].transform(X_test)
        
        # Create and train models
        models = self.create_ensemble_models()
        trained_models = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Training {model_name}...")
            
            # Train model
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
            
            # Store model and performance
            trained_models[model_name] = model
            self.model_performance[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_columns, model.feature_importances_))
                self.feature_importance[model_name] = importance_dict
            
            self.logger.info(f"   {model_name}: F1={f1:.3f}, AUC={auc:.3f}")
        
        # Create voting ensemble
        self.models['voting_ensemble'] = VotingClassifier(
            estimators=[
                ('gb', trained_models['gradient_boosting']),
                ('xgb', trained_models['xgboost']),
                ('lgb', trained_models['lightgbm']),
                ('rf', trained_models['random_forest'])
            ],
            voting='soft'
        )
        
        # Train ensemble
        self.logger.info("Training voting ensemble...")
        self.models['voting_ensemble'].fit(X_train_scaled, y_train)
        
        # Evaluate ensemble
        y_pred_ensemble = self.models['voting_ensemble'].predict(X_test_scaled)
        y_pred_proba_ensemble = self.models['voting_ensemble'].predict_proba(X_test_scaled)[:, 1]
        
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        ensemble_f1 = f1_score(y_test, y_pred_ensemble, zero_division=0)
        ensemble_auc = roc_auc_score(y_test, y_pred_proba_ensemble)
        
        self.model_performance['voting_ensemble'] = {
            'accuracy': ensemble_accuracy,
            'f1_score': ensemble_f1,
            'auc_score': ensemble_auc,
            'is_ensemble': True
        }
        
        self.logger.info(f"   Voting Ensemble: F1={ensemble_f1:.3f}, AUC={ensemble_auc:.3f}")
        
        # Update individual models
        self.models.update(trained_models)
        
        # Save models
        self.save_models()
        
        self.metrics['model_retrains'] += 1
        self.metrics['last_retrain'] = datetime.now().isoformat()
        
        return {
            'ensemble_performance': self.model_performance['voting_ensemble'],
            'individual_performance': {k: v for k, v in self.model_performance.items() if k != 'voting_ensemble'},
            'feature_importance': self.feature_importance
        }
    
    def predict_risk_score(self, user_features):
        """Predict risk score using ensemble models"""
        if not self.models:
            self.logger.warning("No models trained yet")
            return self.calculate_static_risk_score(user_features)
        
        start_time = time.time()
        
        try:
            # Prepare features
            features_df = pd.DataFrame([user_features])
            features_scaled = self.scalers['feature'].transform(features_df[self.feature_columns])
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for model_name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(features_scaled)[0, 1]
                    probabilities[model_name] = pred_proba
                    predictions[model_name] = model.predict(features_scaled)[0]
            
            # Ensemble prediction (weighted average)
            if 'voting_ensemble' in probabilities:
                ensemble_risk = probabilities['voting_ensemble']
            else:
                # Weighted average of individual models
                weights = {
                    'gradient_boosting': 0.25,
                    'xgboost': 0.25,
                    'lightgbm': 0.25,
                    'random_forest': 0.25
                }
                
                ensemble_risk = sum(
                    probabilities.get(model, 0) * weight 
                    for model, weight in weights.items()
                    if model in probabilities
                )
            
            # Calculate prediction time
            prediction_time = (time.time() - start_time) * 1000
            
            # Store prediction
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'risk_score': ensemble_risk,
                'individual_predictions': probabilities,
                'prediction_time_ms': prediction_time,
                'features': user_features
            }
            
            self.prediction_history.append(prediction_record)
            self.metrics['total_predictions'] += 1
            
            if ensemble_risk >= self.risk_thresholds['high']:
                self.metrics['high_risk_predictions'] += 1
            
            # Update average prediction time
            recent_times = [p['prediction_time_ms'] for p in list(self.prediction_history)[-100:]]
            self.metrics['avg_prediction_time_ms'] = np.mean(recent_times)
            
            return {
                'risk_score': ensemble_risk,
                'risk_level': self.get_risk_level(ensemble_risk),
                'confidence': self.calculate_prediction_confidence(probabilities),
                'individual_predictions': probabilities,
                'prediction_time_ms': prediction_time,
                'model_used': 'ensemble'
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting risk score: {str(e)}")
            return self.calculate_static_risk_score(user_features)
    
    def calculate_static_risk_score(self, user_features):
        """Fallback static risk score calculation"""
        risk_factors = []
        
        # Performance-based risk
        if user_features.get('accuracy', 0) > 90:
            risk_factors.append(0.3)
        if user_features.get('kills_per_game', 0) > 20:
            risk_factors.append(0.25)
        if user_features.get('win_rate', 0) > 0.8:
            risk_factors.append(0.2)
        
        # Behavioral risk
        if user_features.get('login_frequency', 0) > 10:
            risk_factors.append(0.15)
        if user_features.get('performance_consistency', 1) > 0.95:
            risk_factors.append(0.2)
        
        # Network risk
        if user_features.get('connected_accounts_count', 0) > 5:
            risk_factors.append(0.1)
        
        risk_score = min(1.0, sum(risk_factors))
        
        return {
            'risk_score': risk_score,
            'risk_level': self.get_risk_level(risk_score),
            'confidence': 0.5,
            'model_used': 'static'
        }
    
    def get_risk_level(self, risk_score):
        """Get risk level classification"""
        if risk_score >= self.risk_thresholds['critical']:
            return "CRITICAL"
        elif risk_score >= self.risk_thresholds['high']:
            return "HIGH"
        elif risk_score >= self.risk_thresholds['medium']:
            return "MEDIUM"
        elif risk_score >= self.risk_thresholds['low']:
            return "LOW"
        else:
            return "MINIMAL"
    
    def calculate_prediction_confidence(self, probabilities):
        """Calculate confidence in prediction based on model agreement"""
        if not probabilities:
            return 0.5
        
        # Calculate standard deviation of predictions
        pred_values = list(probabilities.values())
        std_dev = np.std(pred_values)
        
        # Lower std dev = higher confidence
        confidence = max(0, 1 - std_dev)
        return confidence
    
    def update_model_with_feedback(self, user_features, actual_outcome):
        """Update models with new feedback (online learning)"""
        feedback_record = {
            'timestamp': datetime.now().isoformat(),
            'features': user_features,
            'actual_outcome': actual_outcome,
            'predicted_risk': self.predict_risk_score(user_features)['risk_score']
        }
        
        self.feedback_data.append(feedback_record)
        
        # Retrain models if we have enough feedback
        if len(self.feedback_data) >= 100:
            self.logger.info("Retraining models with new feedback...")
            
            # Convert feedback to training data
            feedback_df = pd.DataFrame(list(self.feedback_data))
            feedback_df['is_high_risk'] = feedback_df['actual_outcome']
            
            # Retrain
            self.train_ensemble_models(feedback_df)
            
            # Clear feedback data
            self.feedback_data.clear()
            
            self.metrics['accuracy_improvements'] += 1
    
    def optimize_risk_thresholds(self):
        """Optimize risk thresholds based on prediction history"""
        if len(self.prediction_history) < 1000:
            return
        
        self.logger.info("Optimizing risk thresholds...")
        
        # Extract risk scores and actual outcomes (if available)
        risk_scores = [p['risk_score'] for p in self.prediction_history]
        
        # Use statistical methods to find optimal thresholds
        percentiles = [20, 40, 60, 80, 95]
        new_thresholds = {}
        
        for i, percentile in enumerate(percentiles):
            threshold_value = np.percentile(risk_scores, percentile)
            
            if i == 0:
                new_thresholds['minimal'] = threshold_value
            elif i == 1:
                new_thresholds['low'] = threshold_value
            elif i == 2:
                new_thresholds['medium'] = threshold_value
            elif i == 3:
                new_thresholds['high'] = threshold_value
            else:
                new_thresholds['critical'] = min(1.0, threshold_value)
        
        # Update thresholds
        self.risk_thresholds = new_thresholds
        
        self.logger.info(f"Updated risk thresholds: {new_thresholds}")
    
    def save_models(self):
        """Save trained models and scalers"""
        models_dir = os.path.join(self.production_path, "risk_models")
        os.makedirs(models_dir, exist_ok=True)
        
        try:
            # Save individual models
            for model_name, model in self.models.items():
                if model_name != 'voting_ensemble':  # Skip voting ensemble for now
                    model_path = os.path.join(models_dir, f"{model_name}.pkl")
                    joblib.dump(model, model_path)
            
            # Save scalers
            for scaler_name, scaler in self.scalers.items():
                scaler_path = os.path.join(models_dir, f"{scaler_name}_scaler.pkl")
                joblib.dump(scaler, scaler_path)
            
            # Save thresholds
            thresholds_path = os.path.join(models_dir, "risk_thresholds.json")
            with open(thresholds_path, 'w') as f:
                json.dump(self.risk_thresholds, f, indent=2)
            
            # Save performance metrics
            performance_path = os.path.join(models_dir, "model_performance.json")
            with open(performance_path, 'w') as f:
                json.dump(self.model_performance, f, indent=2)
            
            self.logger.info("Models and components saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': self.metrics,
            'model_performance': self.model_performance,
            'risk_thresholds': self.risk_thresholds,
            'feature_importance': self.feature_importance,
            'prediction_statistics': self.calculate_prediction_statistics(),
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def calculate_prediction_statistics(self):
        """Calculate prediction statistics"""
        if not self.prediction_history:
            return {}
        
        risk_scores = [p['risk_score'] for p in self.prediction_history]
        prediction_times = [p['prediction_time_ms'] for p in self.prediction_history]
        
        return {
            'total_predictions': len(risk_scores),
            'avg_risk_score': np.mean(risk_scores),
            'max_risk_score': np.max(risk_scores),
            'min_risk_score': np.min(risk_scores),
            'risk_score_std': np.std(risk_scores),
            'avg_prediction_time_ms': np.mean(prediction_times),
            'max_prediction_time_ms': np.max(prediction_times),
            'risk_distribution': self.calculate_risk_distribution(risk_scores)
        }
    
    def calculate_risk_distribution(self, risk_scores):
        """Calculate distribution of risk scores"""
        distribution = {
            'minimal': sum(1 for s in risk_scores if s < self.risk_thresholds['low']),
            'low': sum(1 for s in risk_scores if self.risk_thresholds['low'] <= s < self.risk_thresholds['medium']),
            'medium': sum(1 for s in risk_scores if self.risk_thresholds['medium'] <= s < self.risk_thresholds['high']),
            'high': sum(1 for s in risk_scores if self.risk_thresholds['high'] <= s < self.risk_thresholds['critical']),
            'critical': sum(1 for s in risk_scores if s >= self.risk_thresholds['critical'])
        }
        
        total = len(risk_scores)
        for level in distribution:
            distribution[level] = distribution[level] / total if total > 0 else 0
        
        return distribution
    
    def generate_recommendations(self):
        """Generate system recommendations"""
        recommendations = []
        
        # Model performance recommendations
        if self.model_performance:
            best_f1 = max([p.get('f1_score', 0) for p in self.model_performance.values()])
            if best_f1 < 0.8:
                recommendations.append("Consider collecting more training data to improve model accuracy")
            
            if self.metrics['avg_prediction_time_ms'] > 50:
                recommendations.append("Model prediction time is high - consider model optimization")
        
        # Data recommendations
        if len(self.feedback_data) < 50:
            recommendations.append("Collect more user feedback for continuous improvement")
        
        # Threshold recommendations
        high_risk_ratio = self.metrics['high_risk_predictions'] / max(1, self.metrics['total_predictions'])
        if high_risk_ratio > 0.3:
            recommendations.append("High percentage of high-risk predictions - review risk thresholds")
        
        return recommendations

if __name__ == "__main__":
    print("🎯 STELLOR LOGIC AI - ADVANCED RISK SCORING ENGINE")
    print("=" * 60)
    print("Gradient boosting ensemble with dynamic learning")
    print("=" * 60)
    
    # Install required packages
    try:
        import xgboost
        import lightgbm
        import joblib
    except ImportError:
        print("📦 Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "xgboost", "lightgbm", "joblib"])
        import xgboost
        import lightgbm
        import joblib
    
    # Initialize risk scoring engine
    risk_engine = AdvancedRiskScoringEngine()
    
    try:
        # Train ensemble models
        print("🤖 Training ensemble risk scoring models...")
        training_results = risk_engine.train_ensemble_models()
        
        print(f"\n📊 TRAINING RESULTS:")
        ensemble_perf = training_results['ensemble_performance']
        print(f"   Ensemble Accuracy: {ensemble_perf['accuracy']:.3f}")
        print(f"   Ensemble F1-Score: {ensemble_perf['f1_score']:.3f}")
        print(f"   Ensemble AUC: {ensemble_perf['auc_score']:.3f}")
        
        print(f"\n🎯 INDIVIDUAL MODEL PERFORMANCE:")
        for model_name, perf in training_results['individual_performance'].items():
            print(f"   {model_name}: F1={perf['f1_score']:.3f}, AUC={perf['auc_score']:.3f}")
        
        # Test predictions
        print(f"\n🔍 Testing risk predictions...")
        
        test_users = [
            {
                'accuracy': 95,
                'kills_per_game': 25,
                'deaths_per_game': 2,
                'win_rate': 0.9,
                'avg_session_length': 180,
                'login_frequency': 12,
                'playtime_variance': 5000,
                'performance_consistency': 0.98,
                'connected_accounts_count': 8,
                'account_age_days': 15,
                'peak_hours_count': 6,
                'weekend_ratio': 0.8,
                'recent_anomaly_count': 5,
                'cross_account_similarity': 0.9,
                'device_fingerprint_score': 0.85,
                'ip_risk_score': 0.7
            },
            {
                'accuracy': 45,
                'kills_per_game': 8,
                'deaths_per_game': 10,
                'win_rate': 0.4,
                'avg_session_length': 90,
                'login_frequency': 2,
                'playtime_variance': 1000,
                'performance_consistency': 0.7,
                'connected_accounts_count': 1,
                'account_age_days': 300,
                'peak_hours_count': 2,
                'weekend_ratio': 0.3,
                'recent_anomaly_count': 0,
                'cross_account_similarity': 0.2,
                'device_fingerprint_score': 0.3,
                'ip_risk_score': 0.1
            }
        ]
        
        for i, user_features in enumerate(test_users):
            result = risk_engine.predict_risk_score(user_features)
            user_type = "High Risk" if i == 0 else "Low Risk"
            print(f"   Test User {i+1} ({user_type}):")
            print(f"      Risk Score: {result['risk_score']:.3f}")
            print(f"      Risk Level: {result['risk_level']}")
            print(f"      Confidence: {result['confidence']:.3f}")
            print(f"      Prediction Time: {result['prediction_time_ms']:.2f}ms")
        
        # Optimize thresholds
        print(f"\n⚙️ Optimizing risk thresholds...")
        risk_engine.optimize_risk_thresholds()
        
        # Generate performance report
        print(f"\n📋 Generating performance report...")
        report = risk_engine.generate_performance_report()
        
        print(f"\n📊 SYSTEM PERFORMANCE:")
        stats = report['prediction_statistics']
        print(f"   Total Predictions: {stats.get('total_predictions', 0)}")
        print(f"   Average Risk Score: {stats.get('avg_risk_score', 0):.3f}")
        print(f"   Average Prediction Time: {stats.get('avg_prediction_time_ms', 0):.2f}ms")
        print(f"   Model Retrains: {report['system_metrics']['model_retrains']}")
        
        print(f"\n🎉 ADVANCED RISK SCORING SUCCESSFUL!")
        print(f"✅ Ensemble models trained and operational")
        print(f"✅ Dynamic risk scoring implemented")
        print(f"✅ Real-time predictions working")
        print(f"✅ Continuous learning framework ready")
        print(f"✅ Performance optimization active")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
