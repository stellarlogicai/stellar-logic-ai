"""
Stellar Logic AI - Anti-Cheat Detection System
Adapted gaming anti-cheat techniques for AI behavioral analysis
Detects suspicious patterns, anomalies, and behavioral deviations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import hashlib
import json

@dataclass
class BehavioralSignature:
    """Represents a unique behavioral signature for pattern matching"""
    signature_id: str
    pattern_type: str
    characteristics: Dict
    confidence_score: float
    first_seen: datetime
    last_seen: datetime
    frequency: int

class AntiCheatDetector:
    """
    Core anti-cheat detection system adapted for AI behavioral analysis
    Uses gaming anti-cheat techniques to detect suspicious AI company behaviors
    """
    
    def __init__(self):
        self.behavioral_signatures = {}
        self.anomaly_thresholds = self._initialize_thresholds()
        self.pattern_history = defaultdict(deque)
        self.detection_algorithms = {
            'statistical_anomaly': self._detect_statistical_anomalies,
            'pattern_deviation': self._detect_pattern_deviations,
            'temporal_inconsistency': self._detect_temporal_inconsistencies,
            'behavioral_correlation': self._detect_behavioral_correlations
        }
        
    def _initialize_thresholds(self) -> Dict:
        """Initialize detection thresholds for different behavioral patterns"""
        return {
            'timing_variance': 2.5,  # Standard deviations
            'content_similarity': 0.85,  # Similarity threshold
            'frequency_spike': 3.0,  # Frequency multiplier
            'pattern_deviation': 0.7,  # Deviation threshold
            'temporal_gap': 3600  # Seconds
        }
    
    def analyze_behavioral_patterns(self, company_data: List[Dict]) -> Dict:
        """
        Analyze behavioral patterns using anti-cheat detection algorithms
        Returns comprehensive analysis of suspicious behaviors
        """
        analysis_results = {
            'company_id': company_data[0].get('company_id', 'unknown'),
            'analysis_timestamp': datetime.now().isoformat(),
            'total_items_analyzed': len(company_data),
            'detection_results': {},
            'risk_assessment': {},
            'behavioral_signatures': [],
            'recommendations': []
        }
        
        # Run all detection algorithms
        for algorithm_name, algorithm_func in self.detection_algorithms.items():
            try:
                result = algorithm_func(company_data)
                analysis_results['detection_results'][algorithm_name] = result
            except Exception as e:
                analysis_results['detection_results'][algorithm_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Calculate overall risk assessment
        analysis_results['risk_assessment'] = self._calculate_risk_assessment(
            analysis_results['detection_results']
        )
        
        # Generate behavioral signatures
        analysis_results['behavioral_signatures'] = self._generate_behavioral_signatures(
            company_data
        )
        
        # Generate recommendations
        analysis_results['recommendations'] = self._generate_security_recommendations(
            analysis_results['risk_assessment']
        )
        
        return analysis_results
    
    def _detect_statistical_anomalies(self, data: List[Dict]) -> Dict:
        """Detect statistical anomalies in behavioral patterns"""
        anomalies = []
        
        # Extract numerical features for analysis
        timestamps = [datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')) for item in data]
        content_lengths = [len(item.get('content', '')) for item in data]
        
        # Detect timing anomalies
        if len(timestamps) > 1:
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            mean_diff = np.mean(time_diffs)
            std_diff = np.std(time_diffs)
            
            for i, diff in enumerate(time_diffs):
                z_score = abs(diff - mean_diff) / std_diff if std_diff > 0 else 0
                if z_score > self.anomaly_thresholds['timing_variance']:
                    anomalies.append({
                        'type': 'timing_anomaly',
                        'severity': 'high' if z_score > 3 else 'medium',
                        'z_score': z_score,
                        'timestamp': timestamps[i+1].isoformat(),
                        'description': f'Unusual timing pattern detected (Z-score: {z_score:.2f})'
                    })
        
        # Detect content length anomalies
        if len(content_lengths) > 1:
            mean_length = np.mean(content_lengths)
            std_length = np.std(content_lengths)
            
            for i, length in enumerate(content_lengths):
                z_score = abs(length - mean_length) / std_length if std_length > 0 else 0
                if z_score > self.anomaly_thresholds['timing_variance']:
                    anomalies.append({
                        'type': 'content_anomaly',
                        'severity': 'medium' if z_score > 2 else 'low',
                        'z_score': z_score,
                        'timestamp': timestamps[i].isoformat(),
                        'description': f'Unusual content length detected (Z-score: {z_score:.2f})'
                    })
        
        return {
            'anomalies_detected': len(anomalies),
            'anomaly_details': anomalies,
            'statistical_summary': {
                'mean_time_diff': mean_diff if len(time_diffs) > 0 else 0,
                'std_time_diff': std_diff if len(time_diffs) > 0 else 0,
                'mean_content_length': mean_length,
                'std_content_length': std_length
            }
        }
    
    def _detect_pattern_deviations(self, data: List[Dict]) -> Dict:
        """Detect deviations from expected behavioral patterns"""
        deviations = []
        
        # Analyze content patterns
        content_patterns = self._extract_content_patterns(data)
        expected_patterns = self._get_expected_patterns()
        
        for pattern_type, pattern_data in content_patterns.items():
            if pattern_type in expected_patterns:
                expected = expected_patterns[pattern_type]
                actual = pattern_data
                
                deviation_score = self._calculate_pattern_deviation(expected, actual)
                
                if deviation_score > self.anomaly_thresholds['pattern_deviation']:
                    deviations.append({
                        'type': 'pattern_deviation',
                        'pattern_type': pattern_type,
                        'deviation_score': deviation_score,
                        'severity': 'high' if deviation_score > 0.8 else 'medium',
                        'description': f'Significant deviation in {pattern_type} pattern'
                    })
        
        # Check for unusual content similarity
        similarities = self._calculate_content_similarities(data)
        high_similarity_pairs = [(i, j, sim) for i, j, sim in similarities if sim > self.anomaly_thresholds['content_similarity']]
        
        for i, j, similarity in high_similarity_pairs:
            deviations.append({
                'type': 'content_similarity',
                'severity': 'medium',
                'similarity_score': similarity,
                'item_indices': [i, j],
                'description': f'Unusually high content similarity ({similarity:.2f}) detected'
            })
        
        return {
            'deviations_detected': len(deviations),
            'deviation_details': deviations,
            'pattern_analysis': content_patterns
        }
    
    def _detect_temporal_inconsistencies(self, data: List[Dict]) -> Dict:
        """Detect temporal inconsistencies in behavioral patterns"""
        inconsistencies = []
        
        timestamps = [datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')) for item in data]
        
        # Check for unusual time gaps
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).total_seconds()
            
            if gap > self.anomaly_thresholds['temporal_gap']:
                inconsistencies.append({
                    'type': 'temporal_gap',
                    'gap_seconds': gap,
                    'severity': 'high' if gap > 7200 else 'medium',
                    'start_time': timestamps[i-1].isoformat(),
                    'end_time': timestamps[i].isoformat(),
                    'description': f'Unusual temporal gap of {gap/3600:.1f} hours detected'
                })
        
        # Check for clustering (unusual concentration of activity)
        if len(timestamps) > 3:
            for i in range(len(timestamps) - 2):
                window_start = timestamps[i]
                window_end = timestamps[i + 2]
                window_duration = (window_end - window_start).total_seconds()
                
                if window_duration < 3600:  # 3 items within 1 hour
                    inconsistencies.append({
                        'type': 'activity_clustering',
                        'window_duration': window_duration,
                        'items_in_window': 3,
                        'severity': 'medium',
                        'description': f'Unusual activity clustering detected'
                    })
        
        return {
            'inconsistencies_detected': len(inconsistencies),
            'inconsistency_details': inconsistencies,
            'temporal_analysis': {
                'total_timespan': (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 0,
                'average_gap': np.mean([(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]) if len(timestamps) > 1 else 0
            }
        }
    
    def _detect_behavioral_correlations(self, data: List[Dict]) -> Dict:
        """Detect correlations between different behavioral patterns"""
        correlations = []
        
        # Extract behavioral features
        features = []
        for item in data:
            feature_vector = {
                'hour_of_day': datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')).hour,
                'content_length': len(item.get('content', '')),
                'has_competitor_mentions': any(comp in item.get('content', '').lower() for comp in ['openai', 'anthropic', 'google', 'microsoft']),
                'urgency_level': len([word for word in ['urgent', 'immediate', 'breaking', 'critical'] if word in item.get('content', '').lower()]),
                'technical_complexity': len([word for word in ['api', 'algorithm', 'model', 'neural', 'transformer'] if word in item.get('content', '').lower()])
            }
            features.append(feature_vector)
        
        # Calculate correlation matrix
        if len(features) > 1:
            df = pd.DataFrame(features)
            correlation_matrix = df.corr()
            
            # Find strong correlations
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # Strong correlation threshold
                        correlations.append({
                            'feature_1': correlation_matrix.columns[i],
                            'feature_2': correlation_matrix.columns[j],
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate',
                            'description': f'Behavioral correlation between {correlation_matrix.columns[i]} and {correlation_matrix.columns[j]}'
                        })
        
        return {
            'correlations_detected': len(correlations),
            'correlation_details': correlations,
            'feature_analysis': {
                'total_features': len(features[0]) if features else 0,
                'data_points': len(features)
            }
        }
    
    def _extract_content_patterns(self, data: List[Dict]) -> Dict:
        """Extract patterns from content analysis"""
        patterns = {
            'timing_patterns': [],
            'content_themes': [],
            'urgency_patterns': [],
            'competitor_mentions': []
        }
        
        for item in data:
            timestamp = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
            content = item.get('content', '').lower()
            
            # Timing patterns
            patterns['timing_patterns'].append(timestamp.hour)
            
            # Content themes
            if 'launch' in content or 'release' in content:
                patterns['content_themes'].append('product_launch')
            elif 'funding' in content or 'investment' in content:
                patterns['content_themes'].append('funding_activity')
            elif 'partnership' in content or 'collaboration' in content:
                patterns['content_themes'].append('partnership_activity')
            
            # Urgency patterns
            urgency_words = ['urgent', 'immediate', 'breaking', 'critical']
            if any(word in content for word in urgency_words):
                patterns['urgency_patterns'].append(1)
            else:
                patterns['urgency_patterns'].append(0)
            
            # Competitor mentions
            competitors = ['openai', 'anthropic', 'google', 'microsoft', 'meta', 'amazon']
            mentioned = [comp for comp in competitors if comp in content]
            patterns['competitor_mentions'].extend(mentioned)
        
        return patterns
    
    def _get_expected_patterns(self) -> Dict:
        """Get expected behavioral patterns for comparison"""
        return {
            'timing_patterns': {
                'mean': 14.0,  # 2 PM average
                'std': 4.0
            },
            'content_themes': {
                'distribution': {'product_launch': 0.3, 'funding_activity': 0.2, 'partnership_activity': 0.2, 'other': 0.3}
            },
            'urgency_patterns': {
                'mean': 0.2,
                'std': 0.1
            }
        }
    
    def _calculate_pattern_deviation(self, expected: Dict, actual) -> float:
        """Calculate deviation score between expected and actual patterns"""
        if isinstance(actual, list) and len(actual) > 0:
            if 'mean' in expected:
                actual_mean = np.mean(actual)
                actual_std = np.std(actual)
                
                # Calculate deviation score
                mean_deviation = abs(actual_mean - expected['mean']) / expected['mean']
                std_deviation = abs(actual_std - expected.get('std', 0)) / max(expected.get('std', 1), 1)
                
                return (mean_deviation + std_deviation) / 2
        
        return 0.5  # Default deviation
    
    def _calculate_content_similarities(self, data: List[Dict]) -> List[Tuple[int, int, float]]:
        """Calculate similarity scores between content items"""
        similarities = []
        
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                content1 = data[i].get('content', '').lower()
                content2 = data[j].get('content', '').lower()
                
                # Simple similarity calculation (can be enhanced with NLP)
                similarity = self._text_similarity(content1, content2)
                similarities.append((i, j, similarity))
        
        return similarities
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
    
    def _calculate_risk_assessment(self, detection_results: Dict) -> Dict:
        """Calculate overall risk assessment from detection results"""
        risk_scores = []
        total_anomalies = 0
        high_severity_count = 0
        
        for algorithm, results in detection_results.items():
            if isinstance(results, dict) and 'error' not in results:
                # Count anomalies and issues
                if 'anomalies_detected' in results:
                    total_anomalies += results['anomalies_detected']
                
                if 'deviations_detected' in results:
                    total_anomalies += results['deviations_detected']
                
                if 'inconsistencies_detected' in results:
                    total_anomalies += results['inconsistencies_detected']
                
                # Count high severity issues
                for key in ['anomaly_details', 'deviation_details', 'inconsistency_details']:
                    if key in results:
                        high_severity_count += len([item for item in results[key] if item.get('severity') == 'high'])
        
        # Calculate risk score
        base_risk = min(100, total_anomalies * 5)
        severity_multiplier = 1 + (high_severity_count * 0.2)
        final_risk_score = min(100, base_risk * severity_multiplier)
        
        return {
            'overall_risk_score': round(final_risk_score, 2),
            'risk_level': self._get_risk_level(final_risk_score),
            'total_anomalies': total_anomalies,
            'high_severity_issues': high_severity_count,
            'risk_factors': {
                'anomaly_count': total_anomalies,
                'severity_concentration': high_severity_count / max(total_anomalies, 1),
                'detection_coverage': len([r for r in detection_results.values() if isinstance(r, dict) and 'error' not in r])
            }
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Get risk level based on score"""
        if score >= 80:
            return 'critical'
        elif score >= 60:
            return 'high'
        elif score >= 40:
            return 'medium'
        elif score >= 20:
            return 'low'
        else:
            return 'minimal'
    
    def _generate_behavioral_signatures(self, data: List[Dict]) -> List[Dict]:
        """Generate unique behavioral signatures for pattern matching"""
        signatures = []
        
        # Group by content patterns
        content_groups = defaultdict(list)
        for i, item in enumerate(data):
            content_hash = hashlib.md5(item.get('content', '').encode()).hexdigest()[:8]
            content_groups[content_hash].append(i)
        
        # Generate signatures for repeated patterns
        for content_hash, indices in content_groups.items():
            if len(indices) > 1:  # Repeated pattern
                signature = {
                    'signature_id': f"sig_{content_hash}",
                    'pattern_type': 'repeated_content',
                    'occurrences': len(indices),
                    'first_seen': data[indices[0]]['timestamp'],
                    'last_seen': data[indices[-1]]['timestamp'],
                    'confidence_score': min(1.0, len(indices) / 10),
                    'characteristics': {
                        'content_length': len(data[indices[0]].get('content', '')),
                        'pattern_frequency': len(indices)
                    }
                }
                signatures.append(signature)
        
        return signatures
    
    def _generate_security_recommendations(self, risk_assessment: Dict) -> List[str]:
        """Generate security recommendations based on risk assessment"""
        recommendations = []
        risk_level = risk_assessment['risk_level']
        
        if risk_level in ['critical', 'high']:
            recommendations.append("IMMEDIATE ACTION: Implement enhanced monitoring for this entity")
            recommendations.append("Review all recent activities for potential competitive threats")
        
        if risk_assessment['high_severity_issues'] > 0:
            recommendations.append("Investigate high-severity behavioral anomalies immediately")
        
        if risk_assessment['total_anomalies'] > 10:
            recommendations.append("Consider implementing automated alerting for behavioral deviations")
        
        if risk_level == 'medium':
            recommendations.append("Include in regular behavioral analysis reviews")
        
        return recommendations

# Initialize the anti-cheat detection system
anti_cheat_detector = AntiCheatDetector()

# Example usage
if __name__ == "__main__":
    # Test data
    test_data = [
        {
            'company_id': 'test_company',
            'timestamp': '2024-03-08T14:30:00Z',
            'content': 'Breaking news: We are launching our revolutionary AI system that surpasses all competitors'
        },
        {
            'company_id': 'test_company',
            'timestamp': '2024-03-08T14:35:00Z',
            'content': 'Breaking news: We are launching our revolutionary AI system that surpasses all competitors'
        },
        {
            'company_id': 'test_company',
            'timestamp': '2024-03-08T02:00:00Z',
            'content': 'Urgent announcement regarding our competitive positioning against OpenAI and Microsoft'
        }
    ]
    
    # Run analysis
    result = anti_cheat_detector.analyze_behavioral_patterns(test_data)
    print("Anti-Cheat Detection Results:")
    print(json.dumps(result, indent=2))
