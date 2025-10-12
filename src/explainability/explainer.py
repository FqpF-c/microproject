import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import google.generativeai as genai
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExplainer:
    """
    Generate human-readable explanations for Android malware classification results.
    Integrates with Gemini API for natural language explanations.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gemini_api_key = os.getenv(config.get('gemini_api_key_env', 'AIzaSyCa7kBWIKbj-9rSTveheLWSlFNHWFM_SV0'))
        self.model_name = config.get('model_name', 'gemini-pro')
        self.temperature = config.get('temperature', 0.3)
        self.max_tokens = config.get('max_tokens', 200)
        self.explanation_template = config.get('explanation_template', '')

        # Initialize Gemini API
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel(self.model_name)
            logger.info("Gemini API initialized successfully")
        else:
            logger.warning("GEMINI_API_KEY not found. Explanations will be rule-based only.")
            self.gemini_model = None

        # Feature importance mappings
        self.feature_descriptions = self._load_feature_descriptions()
        self.risk_explanations = self._load_risk_explanations()

    def _load_feature_descriptions(self) -> Dict[str, str]:
        """Load human-readable descriptions for features."""
        return {
            # Permission features
            'dangerous_permissions_count': 'Number of dangerous permissions requested',
            'has_perm_read_sms': 'Can read SMS messages',
            'has_perm_send_sms': 'Can send SMS messages',
            'has_perm_read_contacts': 'Can access contact list',
            'has_perm_access_fine_location': 'Can access precise location',
            'has_perm_record_audio': 'Can record audio',
            'has_perm_camera': 'Can access camera',
            'has_perm_read_phone_state': 'Can read phone information',
            'has_perm_call_phone': 'Can make phone calls',
            'has_perm_system_alert_window': 'Can display system-level windows',
            'has_perm_device_admin': 'Has device administrator privileges',

            # API call features
            'suspicious_api_count': 'Number of suspicious API calls',
            'crypto_api_count': 'Number of cryptographic API calls',
            'network_api_count': 'Number of network-related API calls',
            'sms_api_count': 'Number of SMS-related API calls',

            # Opcode features
            'total_opcodes': 'Total number of bytecode instructions',
            'reflection_opcodes': 'Number of reflection-based calls',

            # Manifest features
            'activities_count': 'Number of app activities',
            'services_count': 'Number of background services',
            'receivers_count': 'Number of broadcast receivers',
            'exported_activities': 'Number of activities exposed to other apps',
            'exported_services': 'Number of services exposed to other apps',

            # String features
            'suspicious_strings': 'Number of suspicious text patterns',
            'url_count': 'Number of embedded URLs',

            # Certificate features
            'is_debug_cert': 'Uses debug/development certificate',
            'is_android_debug': 'Uses Android debug certificate'
        }

    def _load_risk_explanations(self) -> Dict[str, Dict[str, str]]:
        """Load risk-specific explanation templates."""
        return {
            'Low': {
                'general': 'This app appears to be benign with normal behavior patterns.',
                'permissions': 'The app requests only standard permissions for its functionality.',
                'apis': 'API usage patterns are consistent with legitimate applications.',
                'behavior': 'No suspicious behavior patterns detected.'
            },
            'Medium': {
                'general': 'This app shows some concerning characteristics that warrant investigation.',
                'permissions': 'The app requests elevated permissions that could be misused.',
                'apis': 'Some API usage patterns are potentially suspicious.',
                'behavior': 'Behavior patterns suggest possible privacy or security concerns.'
            },
            'High': {
                'general': 'This app exhibits malicious behavior patterns and poses significant security risks.',
                'permissions': 'The app requests dangerous permissions typical of malware.',
                'apis': 'API usage patterns are consistent with malicious applications.',
                'behavior': 'Multiple indicators suggest this is likely malware.'
            }
        }

    def extract_top_features(self, feature_importance: np.ndarray,
                           feature_names: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract top contributing features with their importance scores."""
        # Get indices of top features
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]

        top_features = []
        for idx in top_indices:
            if idx < len(feature_names):
                feature_name = feature_names[idx]
                importance = feature_importance[idx]
                top_features.append((feature_name, importance))

        return top_features

    def generate_rule_based_explanation(self, risk_level: str, top_features: List[Tuple[str, float]],
                                      feature_values: Dict[str, Any]) -> str:
        """Generate rule-based explanation without AI."""
        explanations = []

        # Start with general risk assessment
        general_explanation = self.risk_explanations[risk_level]['general']
        explanations.append(general_explanation)

        # Analyze top features
        for feature_name, importance in top_features[:5]:
            if feature_name in self.feature_descriptions:
                description = self.feature_descriptions[feature_name]
                value = feature_values.get(feature_name, 0)

                if 'perm_' in feature_name and value > 0:
                    explanations.append(f"• {description}")
                elif 'api_count' in feature_name and value > 0:
                    explanations.append(f"• Contains {int(value)} {description.lower()}")
                elif 'suspicious' in feature_name and value > 0:
                    explanations.append(f"• Found {int(value)} {description.lower()}")
                elif value > 0:
                    explanations.append(f"• {description}: {value}")

        # Add specific risk indicators
        if risk_level == 'High':
            high_risk_indicators = []
            if feature_values.get('has_perm_send_sms', 0) > 0:
                high_risk_indicators.append("SMS manipulation capabilities")
            if feature_values.get('has_perm_device_admin', 0) > 0:
                high_risk_indicators.append("Device administrator privileges")
            if feature_values.get('suspicious_api_count', 0) > 10:
                high_risk_indicators.append("Extensive use of suspicious APIs")
            if feature_values.get('is_debug_cert', 0) > 0:
                high_risk_indicators.append("Uses debug certificate (common in malware)")

            if high_risk_indicators:
                explanations.append("Key risk factors:")
                explanations.extend([f"• {indicator}" for indicator in high_risk_indicators])

        return " ".join(explanations)

    def generate_gemini_explanation(self, risk_level: str, top_features: List[Tuple[str, float]],
                                  feature_values: Dict[str, Any]) -> Optional[str]:
        """Generate AI-powered explanation using Gemini API."""
        if not self.gemini_model:
            return None

        try:
            # Prepare feature information for the prompt
            feature_info = []
            for feature_name, importance in top_features[:8]:
                description = self.feature_descriptions.get(feature_name, feature_name)
                value = feature_values.get(feature_name, 0)
                feature_info.append(f"- {description}: {value} (importance: {importance:.3f})")

            # Create detailed prompt
            prompt = f"""
            Analyze this Android app classification result and provide a clear explanation:

            CLASSIFICATION: {risk_level} Risk

            TOP CONTRIBUTING FEATURES:
            {chr(10).join(feature_info)}

            Please provide a concise, non-technical explanation (2-3 sentences) that:
            1. Explains why this app is classified as {risk_level} risk
            2. Highlights the most concerning behaviors or permissions
            3. Uses language that a non-technical user can understand
            4. Focuses on security and privacy implications

            Keep the explanation under 150 words and avoid technical jargon.
            """

            # Generate explanation
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )

            if response and response.text:
                return response.text.strip()

        except Exception as e:
            logger.error(f"Error generating Gemini explanation: {e}")

        return None

    def explain_prediction(self, risk_level: str, probabilities: np.ndarray,
                          feature_importance: np.ndarray, feature_names: List[str],
                          feature_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction.

        Args:
            risk_level: Predicted risk level ('Low', 'Medium', 'High')
            probabilities: Class probabilities [P(Low), P(Medium), P(High)]
            feature_importance: Feature importance scores
            feature_names: List of feature names
            feature_values: Dictionary of feature values

        Returns:
            Dictionary containing various explanations and analysis
        """
        logger.info(f"Generating explanation for {risk_level} risk prediction...")

        # Extract top features
        top_features = self.extract_top_features(feature_importance, feature_names)

        # Generate explanations
        rule_based_explanation = self.generate_rule_based_explanation(
            risk_level, top_features, feature_values
        )

        gemini_explanation = self.generate_gemini_explanation(
            risk_level, top_features, feature_values
        )

        # Create comprehensive explanation
        explanation = {
            'risk_level': risk_level,
            'confidence': float(np.max(probabilities)),
            'probabilities': {
                'Low': float(probabilities[0]),
                'Medium': float(probabilities[1]),
                'High': float(probabilities[2])
            },
            'top_features': [
                {
                    'name': name,
                    'importance': float(importance),
                    'description': self.feature_descriptions.get(name, name),
                    'value': feature_values.get(name, 0)
                }
                for name, importance in top_features[:10]
            ],
            'explanations': {
                'rule_based': rule_based_explanation,
                'ai_generated': gemini_explanation,
                'primary': gemini_explanation if gemini_explanation else rule_based_explanation
            },
            'risk_factors': self._analyze_risk_factors(feature_values),
            'recommendations': self._generate_recommendations(risk_level, feature_values),
            'timestamp': datetime.now().isoformat()
        }

        return explanation

    def _analyze_risk_factors(self, feature_values: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze specific risk factors present in the app."""
        risk_factors = {
            'permissions': [],
            'apis': [],
            'behavior': [],
            'certificate': []
        }

        # Permission risks - check for actual dangerous permissions
        dangerous_perms = [
            ('has_perm_send_sms', 'Can send SMS messages'),
            ('has_perm_read_sms', 'Can read SMS messages'),
            ('has_perm_device_admin', 'Has device administrator privileges'),
            ('has_perm_system_alert_window', 'Can display system overlays'),
            ('has_perm_call_phone', 'Can make phone calls'),
            ('has_perm_read_phone_state', 'Can read phone information'),
            ('has_perm_camera', 'Can access camera'),
            ('has_perm_record_audio', 'Can record audio'),
            ('has_perm_read_contacts', 'Can access contact list'),
            ('has_perm_write_external_storage', 'Can write to external storage')
        ]

        for perm_key, description in dangerous_perms:
            if feature_values.get(perm_key, 0) > 0:
                risk_factors['permissions'].append(description)

        # Check for high permission counts
        dangerous_count = feature_values.get('dangerous_permissions_count', 0)
        if dangerous_count > 3:
            risk_factors['permissions'].append(f"Requests {dangerous_count} dangerous permissions")

        # API risks - use available opcode/reflection features
        reflection_count = feature_values.get('reflection_opcodes', 0)
        if reflection_count > 10000:
            risk_factors['apis'].append(f"Heavy use of reflection APIs ({reflection_count:,} calls)")

        total_opcodes = feature_values.get('total_opcodes', 0)
        if total_opcodes > 500000:
            risk_factors['apis'].append(f"Very large codebase ({total_opcodes:,} instructions)")

        # Behavioral risks
        activities_count = feature_values.get('activities_count', 0)
        if activities_count > 10:
            risk_factors['behavior'].append(f"Many app activities ({activities_count})")

        services_count = feature_values.get('services_count', 0)
        if services_count > 5:
            risk_factors['behavior'].append(f"Multiple background services ({services_count})")

        # Check for exported components (exposed to other apps)
        exported_activities = feature_values.get('exported_activities', 0)
        if exported_activities > 0:
            risk_factors['behavior'].append(f"Exposes {exported_activities} activities to other apps")

        # Certificate risks - check for common malware indicators
        file_size = feature_values.get('file_size', 0)
        if file_size > 50000000:  # 50MB
            risk_factors['certificate'].append(f"Unusually large app size ({file_size / 1024 / 1024:.1f}MB)")

        # Check for high opcode counts that might indicate obfuscation
        invoke_virtual_count = feature_values.get('opcode_invoke_virtual', 0)
        if invoke_virtual_count > 50000:
            risk_factors['behavior'].append(f"Excessive method calls ({invoke_virtual_count:,})")

        return risk_factors

    def _generate_recommendations(self, risk_level: str, feature_values: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on risk level and features."""
        recommendations = []

        if risk_level == 'High':
            recommendations.extend([
                "❌ Do not install or immediately uninstall this app",
                "🔒 Check if this app was downloaded from a trusted source",
                "🛡️ Run a full device security scan",
                "📱 Review and revoke any permissions already granted"
            ])

        elif risk_level == 'Medium':
            recommendations.extend([
                "⚠️ Exercise caution when installing this app",
                "🔍 Carefully review all requested permissions",
                "👀 Monitor app behavior after installation",
                "🔒 Consider using app isolation or sandbox features"
            ])

        else:  # Low risk
            recommendations.extend([
                "✅ App appears safe for installation",
                "📋 Still review permissions as a best practice",
                "🔄 Keep the app updated from official sources"
            ])

        # Add specific recommendations based on features
        if feature_values.get('has_perm_send_sms', 0) > 0:
            recommendations.append("📱 Be cautious of unexpected SMS charges")

        if feature_values.get('has_perm_read_phone_state', 0) > 0:
            recommendations.append("📞 Monitor for unauthorized access to phone information")

        if feature_values.get('has_perm_write_external_storage', 0) > 0:
            recommendations.append("💾 Check for unauthorized file access or data theft")

        if feature_values.get('reflection_opcodes', 0) > 50000:
            recommendations.append("🔍 App uses advanced programming techniques - exercise extra caution")

        if feature_values.get('dangerous_permissions_count', 0) > 5:
            recommendations.append("⚠️ App requests many sensitive permissions - review carefully")

        return recommendations

    def batch_explain(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate explanations for multiple predictions."""
        explanations = []

        for i, prediction in enumerate(predictions):
            logger.info(f"Processing explanation {i+1}/{len(predictions)}")

            explanation = self.explain_prediction(
                prediction['risk_level'],
                prediction['probabilities'],
                prediction['feature_importance'],
                prediction['feature_names'],
                prediction['feature_values']
            )

            explanations.append(explanation)

            # Rate limiting for API calls
            if self.gemini_model and i < len(predictions) - 1:
                time.sleep(0.5)  # Avoid API rate limits

        return explanations

    def save_explanation(self, explanation: Dict[str, Any], filepath: str):
        """Save explanation to file."""
        with open(filepath, 'w') as f:
            json.dump(explanation, f, indent=2, default=str)

        logger.info(f"Explanation saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    config = {
        'gemini_api_key_env': 'GEMINI_API_KEY',
        'model_name': 'gemini-pro',
        'temperature': 0.3,
        'max_tokens': 200
    }

    explainer = FeatureExplainer(config)

    # Example prediction
    risk_level = 'High'
    probabilities = np.array([0.1, 0.2, 0.7])
    feature_importance = np.random.random(20)
    feature_names = ['has_perm_send_sms', 'suspicious_api_count', 'crypto_api_count'] + [f'feature_{i}' for i in range(17)]
    feature_values = {
        'has_perm_send_sms': 1,
        'suspicious_api_count': 15,
        'crypto_api_count': 8,
        'dangerous_permissions_count': 5
    }

    # Generate explanation
    explanation = explainer.explain_prediction(
        risk_level, probabilities, feature_importance, feature_names, feature_values
    )

    print(json.dumps(explanation, indent=2, default=str))