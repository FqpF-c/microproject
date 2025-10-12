import os
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional
from collections import Counter
import pandas as pd
import numpy as np
from androguard.misc import AnalyzeAPK
from androguard.core.analysis.analysis import ExternalClass
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APKFeatureExtractor:
    """
    Advanced APK feature extractor using Androguard for static analysis.
    Extracts 100+ features including permissions, API calls, opcodes, manifest data, etc.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_features = config.get('max_features', 150)
        self.include_permissions = config.get('include_permissions', True)
        self.include_api_calls = config.get('include_api_calls', True)
        self.include_opcodes = config.get('include_opcodes', True)
        self.include_manifest = config.get('include_manifest', True)
        self.include_strings = config.get('include_strings', True)
        self.opcode_ngrams = config.get('opcode_ngrams', [1, 2, 3])
        self.api_call_depth = config.get('api_call_depth', 3)

        # Feature vocabularies (will be built during training)
        self.permission_vocab = set()
        self.api_vocab = set()
        self.opcode_vocab = set()
        self.string_vocab = set()

    def analyze_apk(self, apk_path: str) -> Optional[tuple]:
        """Analyze APK file using Androguard."""
        try:
            apk, dalvik_vm_format, dx = AnalyzeAPK(apk_path)
            return apk, dalvik_vm_format, dx
        except Exception as e:
            logger.error(f"Error analyzing APK {apk_path}: {e}")
            return None

    def extract_permissions(self, apk) -> Dict[str, int]:
        """Extract permission features from APK manifest."""
        features = {}

        try:
            permissions = apk.get_permissions()

            # Dangerous permissions
            dangerous_perms = [
                'android.permission.READ_SMS', 'android.permission.SEND_SMS',
                'android.permission.RECEIVE_SMS', 'android.permission.READ_CONTACTS',
                'android.permission.WRITE_CONTACTS', 'android.permission.ACCESS_FINE_LOCATION',
                'android.permission.ACCESS_COARSE_LOCATION', 'android.permission.RECORD_AUDIO',
                'android.permission.CAMERA', 'android.permission.READ_PHONE_STATE',
                'android.permission.CALL_PHONE', 'android.permission.READ_CALL_LOG',
                'android.permission.WRITE_CALL_LOG', 'android.permission.WRITE_EXTERNAL_STORAGE',
                'android.permission.READ_EXTERNAL_STORAGE', 'android.permission.INSTALL_PACKAGES',
                'android.permission.DELETE_PACKAGES', 'android.permission.SYSTEM_ALERT_WINDOW',
                'android.permission.DEVICE_ADMIN', 'android.permission.WAKE_LOCK'
            ]

            # Count dangerous permissions
            features['dangerous_permissions_count'] = sum(1 for perm in permissions if perm in dangerous_perms)
            features['total_permissions_count'] = len(permissions)

            # Individual permission flags
            for perm in dangerous_perms:
                perm_name = perm.split('.')[-1].lower()
                features[f'has_perm_{perm_name}'] = int(perm in permissions)

            # Permission categories
            network_perms = ['INTERNET', 'ACCESS_NETWORK_STATE', 'ACCESS_WIFI_STATE']
            storage_perms = ['READ_EXTERNAL_STORAGE', 'WRITE_EXTERNAL_STORAGE']
            location_perms = ['ACCESS_FINE_LOCATION', 'ACCESS_COARSE_LOCATION']

            features['network_permissions'] = sum(1 for perm in permissions
                                                if any(np in perm for np in network_perms))
            features['storage_permissions'] = sum(1 for perm in permissions
                                                if any(sp in perm for sp in storage_perms))
            features['location_permissions'] = sum(1 for perm in permissions
                                                 if any(lp in perm for lp in location_perms))

        except Exception as e:
            logger.error(f"Error extracting permissions: {e}")

        return features

    def extract_api_calls(self, dx) -> Dict[str, int]:
        """Extract API call features from DEX analysis."""
        features = {}

        try:
            api_calls = []

            # Get all external classes and methods
            for class_analysis in dx.get_classes():
                for method in class_analysis.get_methods():
                    for _, call, _ in method.get_xref_to():
                        if isinstance(call, ExternalClass):
                            api_calls.append(call.get_name())

            # Suspicious API patterns
            suspicious_apis = [
                'sendTextMessage', 'getDeviceId', 'getSubscriberId', 'getLine1Number',
                'getNetworkOperator', 'getSimOperator', 'getLocation', 'execsu',
                'Runtime.exec', 'ProcessBuilder', 'DexClassLoader', 'PathClassLoader',
                'URLClassLoader', 'createFromPdu', 'getInstalledPackages', 'setComponentEnabledSetting'
            ]

            # Count suspicious API usage
            features['suspicious_api_count'] = sum(1 for api in api_calls
                                                 if any(sus_api in api for sus_api in suspicious_apis))

            # Crypto APIs
            crypto_apis = ['Cipher', 'MessageDigest', 'KeyGenerator', 'SecretKey']
            features['crypto_api_count'] = sum(1 for api in api_calls
                                             if any(crypto in api for crypto in crypto_apis))

            # Network APIs
            network_apis = ['HttpURLConnection', 'Socket', 'ServerSocket', 'URL', 'URI']
            features['network_api_count'] = sum(1 for api in api_calls
                                              if any(net in api for net in network_apis))

            # SMS APIs
            sms_apis = ['SmsManager', 'SmsMessage', 'sendMultipartTextMessage']
            features['sms_api_count'] = sum(1 for api in api_calls
                                          if any(sms in api for sms in sms_apis))

            features['total_api_calls'] = len(api_calls)

        except Exception as e:
            logger.error(f"Error extracting API calls: {e}")

        return features

    def extract_opcodes(self, dalvik_vm_format) -> Dict[str, int]:
        """Extract opcode features from Dalvik bytecode."""
        features = {}

        try:
            opcode_counts = Counter()

            for dex in dalvik_vm_format:
                for class_def in dex.get_classes():
                    for method in class_def.get_methods():
                        if method.get_code():
                            for instruction in method.get_code().get_bc().get_instructions():
                                opcode_counts[instruction.get_name()] += 1

            # Most common opcodes
            common_opcodes = [
                'invoke-virtual', 'invoke-direct', 'invoke-static', 'invoke-interface',
                'const', 'const-string', 'const-class', 'new-instance', 'check-cast',
                'instance-of', 'iget', 'iput', 'sget', 'sput', 'move', 'move-result',
                'return', 'return-void', 'if-eq', 'if-ne', 'goto', 'add-int', 'sub-int'
            ]

            total_opcodes = sum(opcode_counts.values())
            features['total_opcodes'] = total_opcodes

            # Opcode frequencies
            for opcode in common_opcodes:
                count = opcode_counts.get(opcode, 0)
                features[f'opcode_{opcode.replace("-", "_")}'] = count
                features[f'opcode_{opcode.replace("-", "_")}_freq'] = count / max(total_opcodes, 1)

            # Suspicious opcode patterns
            reflection_opcodes = ['invoke-virtual', 'invoke-static']
            features['reflection_opcodes'] = sum(opcode_counts.get(op, 0) for op in reflection_opcodes)

        except Exception as e:
            logger.error(f"Error extracting opcodes: {e}")

        return features

    def extract_manifest_features(self, apk) -> Dict[str, int]:
        """Extract features from AndroidManifest.xml."""
        features = {}

        try:
            # Basic manifest info
            features['target_sdk_version'] = apk.get_target_sdk_version() or 0
            features['min_sdk_version'] = apk.get_min_sdk_version() or 0
            features['max_sdk_version'] = apk.get_max_sdk_version() or 0

            # Activities, services, receivers
            features['activities_count'] = len(apk.get_activities())
            features['services_count'] = len(apk.get_services())
            features['receivers_count'] = len(apk.get_receivers())
            features['providers_count'] = len(apk.get_providers())

            # Intent filters
            intent_filters = 0
            for activity in apk.get_activities():
                # This is a simplified check - in real implementation,
                # you'd parse the manifest XML more thoroughly
                intent_filters += 1 if 'MAIN' in str(activity) else 0

            features['intent_filters_count'] = intent_filters

            # Exported components
            features['exported_activities'] = 0  # Would need manifest parsing
            features['exported_services'] = 0
            features['exported_receivers'] = 0

            # Hardware features
            features['uses_camera'] = int('android.hardware.camera' in str(apk.get_android_manifest_xml()))
            features['uses_location'] = int('android.hardware.location' in str(apk.get_android_manifest_xml()))

        except Exception as e:
            logger.error(f"Error extracting manifest features: {e}")

        return features

    def extract_string_features(self, apk) -> Dict[str, int]:
        """Extract string-based features from APK."""
        features = {}

        try:
            strings = []

            # Get strings from DEX
            for string in apk.get_strings():
                strings.append(string)

            # Suspicious string patterns
            suspicious_patterns = [
                'http://', 'https://', 'ftp://', 'ssh://', 'telnet://',
                'su', 'root', '/system/bin/', '/system/app/',
                'android.intent.action.BOOT_COMPLETED',
                'android.provider.Telephony.SMS_RECEIVED',
                'java.lang.Runtime', 'java.lang.ProcessBuilder'
            ]

            features['total_strings'] = len(strings)
            features['suspicious_strings'] = sum(1 for s in strings
                                               if any(pattern in s for pattern in suspicious_patterns))

            # URL patterns
            features['url_count'] = sum(1 for s in strings if 'http' in s.lower())

            # Average string length
            if strings:
                features['avg_string_length'] = np.mean([len(s) for s in strings])
            else:
                features['avg_string_length'] = 0

        except Exception as e:
            logger.error(f"Error extracting string features: {e}")

        return features

    def extract_certificate_features(self, apk) -> Dict[str, int]:
        """Extract certificate-related features."""
        features = {}

        try:
            # Certificate info
            certificate = apk.get_certificate_der(apk.get_certificates()[0])
            cert_sha256 = hashlib.sha256(certificate).hexdigest()

            features['cert_length'] = len(certificate)
            features['is_debug_cert'] = int('debug' in cert_sha256.lower())

            # Well-known debug certificate hash
            debug_cert_hash = "a40da80a59d170caa950cf15c18c454d47a39b26989d8b640ecd745ba71bf5dc"
            features['is_android_debug'] = int(cert_sha256 == debug_cert_hash)

        except Exception as e:
            logger.error(f"Error extracting certificate features: {e}")
            features['cert_length'] = 0
            features['is_debug_cert'] = 0
            features['is_android_debug'] = 0

        return features

    def extract_all_features(self, apk_path: str) -> Dict[str, Any]:
        """Extract all features from an APK file."""
        features = {}

        # Basic file info
        features['file_size'] = os.path.getsize(apk_path)
        features['file_name'] = os.path.basename(apk_path)

        # Analyze APK
        analysis_result = self.analyze_apk(apk_path)
        if not analysis_result:
            return None

        apk, dalvik_vm_format, dx = analysis_result

        # Extract different feature types
        if self.include_permissions:
            features.update(self.extract_permissions(apk))

        if self.include_api_calls:
            features.update(self.extract_api_calls(dx))

        if self.include_opcodes:
            features.update(self.extract_opcodes(dalvik_vm_format))

        if self.include_manifest:
            features.update(self.extract_manifest_features(apk))

        if self.include_strings:
            features.update(self.extract_string_features(apk))

        features.update(self.extract_certificate_features(apk))

        return features

    def extract_dataset_features(self, apk_directory: str, output_path: str) -> pd.DataFrame:
        """Extract features from all APK files in a directory."""
        all_features = []

        apk_files = [f for f in os.listdir(apk_directory) if f.endswith('.apk')]

        logger.info(f"Processing {len(apk_files)} APK files...")

        for i, apk_file in enumerate(apk_files):
            apk_path = os.path.join(apk_directory, apk_file)
            logger.info(f"Processing {i+1}/{len(apk_files)}: {apk_file}")

            features = self.extract_all_features(apk_path)
            if features:
                all_features.append(features)

        # Convert to DataFrame
        df = pd.DataFrame(all_features)

        # Fill missing values
        df = df.fillna(0)

        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")

        return df


if __name__ == "__main__":
    # Example usage
    config = {
        'max_features': 150,
        'include_permissions': True,
        'include_api_calls': True,
        'include_opcodes': True,
        'include_manifest': True,
        'include_strings': True,
        'opcode_ngrams': [1, 2, 3],
        'api_call_depth': 3
    }

    extractor = APKFeatureExtractor(config)

    # Example single APK analysis
    # features = extractor.extract_all_features("path/to/sample.apk")
    # print(json.dumps(features, indent=2))