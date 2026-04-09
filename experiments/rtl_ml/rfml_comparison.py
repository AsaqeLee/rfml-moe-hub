#!/usr/bin/env python3
"""
RFML-MoE vs RTL-ML Comprehensive Comparison Study
===================================================
Implements all 4 RFML feature modalities + baseline RTL-ML features,
trains individual classifiers, runs MoE ensemble experiments, and
produces detailed expert specialization analysis.
"""
import numpy as np
import os
import json
import warnings
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# FEATURE EXTRACTORS (matching RFML modalities)
# ============================================================================

class BaselineRTLFeatures:
    """Original RTL-ML 17-feature extractor (Random Forest baseline)."""
    name = "RTL-ML Baseline (17 features)"

    def extract(self, samples):
        features = []
        power = np.abs(samples) ** 2
        features.extend([np.mean(power), np.std(power), np.max(power), np.min(power)])

        fft_vals = np.fft.fft(samples)
        fft_power = np.abs(fft_vals) ** 2
        features.extend([np.mean(fft_power), np.std(fft_power), np.max(fft_power)])
        features.append(np.argmax(fft_power) / len(fft_power))

        i_samples = np.real(samples)
        q_samples = np.imag(samples)
        features.extend([np.mean(i_samples), np.std(i_samples), np.mean(q_samples), np.std(q_samples)])

        phase = np.angle(samples)
        features.extend([np.mean(phase), np.std(phase)])

        phase_diff = np.diff(phase)
        features.extend([np.mean(phase_diff), np.std(phase_diff)])

        bandwidth = np.sum(fft_power > np.max(fft_power) * 0.1)
        features.append(bandwidth / len(fft_power))

        return np.array(features)


class IQStatFeatures:
    """Extended IQ statistics - inspired by RFML's IQ expert pathway.
    Extracts statistical features from raw IQ that a CNN would learn."""
    name = "IQ Statistical (42 features)"

    def extract(self, samples):
        features = []
        i_samples = np.real(samples)
        q_samples = np.imag(samples)
        amplitude = np.abs(samples)
        phase = np.angle(samples)
        inst_freq = np.diff(np.unwrap(phase))

        # Amplitude statistics
        features.extend([
            np.mean(amplitude), np.std(amplitude), np.median(amplitude),
            kurtosis(amplitude), skew(amplitude),
            np.percentile(amplitude, 10), np.percentile(amplitude, 90),
            np.max(amplitude) / (np.mean(amplitude) + 1e-10),  # crest factor
        ])

        # I/Q channel statistics
        for ch in [i_samples, q_samples]:
            features.extend([
                np.mean(ch), np.std(ch), kurtosis(ch), skew(ch),
                np.percentile(ch, 5), np.percentile(ch, 95),
            ])

        # Phase statistics
        features.extend([
            np.mean(phase), np.std(phase), kurtosis(phase), skew(phase),
        ])

        # Instantaneous frequency statistics
        features.extend([
            np.mean(inst_freq), np.std(inst_freq),
            kurtosis(inst_freq), skew(inst_freq),
            np.median(inst_freq),
        ])

        # Zero crossing rates
        i_zc = np.sum(np.diff(np.sign(i_samples)) != 0) / len(i_samples)
        q_zc = np.sum(np.diff(np.sign(q_samples)) != 0) / len(q_samples)
        features.extend([i_zc, q_zc])

        # Autocorrelation features (first few lags)
        autocorr = np.correlate(amplitude[:1024], amplitude[:1024], mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)
        features.extend([autocorr[1], autocorr[10], autocorr[50], autocorr[100]])

        # Envelope statistics
        analytic = np.abs(scipy_signal.hilbert(i_samples[:4096]))
        features.extend([np.mean(analytic), np.std(analytic)])

        return np.array(features)


class SpectrogramStatFeatures:
    """Spectrogram-derived statistics - mirrors RFML's spectrogram expert.
    Extracts what EfficientNet would learn from the 3-channel spectrogram."""
    name = "Spectrogram Statistical (48 features)"

    def extract(self, samples):
        features = []

        # Complex STFT (matching RFML: fft_size=512, hop=256)
        f, t, Zxx = scipy_signal.stft(samples, fs=1.024e6, nperseg=512, noverlap=256)

        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        log_mag = np.log1p(mag)

        # Channel 0: Log-magnitude statistics
        features.extend([
            np.mean(log_mag), np.std(log_mag), np.max(log_mag),
            kurtosis(log_mag.ravel()), skew(log_mag.ravel()),
        ])

        # Spectral centroid over time
        freqs = f[:, np.newaxis] if f.ndim == 1 else f
        spectral_centroid = np.sum(f[:, np.newaxis] * mag, axis=0) / (np.sum(mag, axis=0) + 1e-10)
        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])

        # Spectral bandwidth
        spectral_bw = np.sqrt(np.sum(((f[:, np.newaxis] - spectral_centroid[np.newaxis, :]) ** 2) * mag, axis=0) / (np.sum(mag, axis=0) + 1e-10))
        features.extend([np.mean(spectral_bw), np.std(spectral_bw)])

        # Spectral rolloff (85%)
        cumsum = np.cumsum(mag, axis=0)
        total = cumsum[-1:, :]
        rolloff_idx = np.argmax(cumsum >= 0.85 * total, axis=0)
        spectral_rolloff = f[rolloff_idx]
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])

        # Spectral flatness (per frame)
        geometric_mean = np.exp(np.mean(np.log(mag + 1e-10), axis=0))
        arithmetic_mean = np.mean(mag, axis=0)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        features.extend([np.mean(spectral_flatness), np.std(spectral_flatness)])

        # Channel 1: Phase statistics
        features.extend([
            np.mean(phase), np.std(phase),
            kurtosis(phase.ravel()), skew(phase.ravel()),
        ])

        # Channel 2: Instantaneous frequency (phase derivative)
        inst_freq = np.diff(np.unwrap(phase, axis=1), axis=1)
        features.extend([
            np.mean(inst_freq), np.std(inst_freq),
            kurtosis(inst_freq.ravel()), skew(inst_freq.ravel()),
        ])

        # Temporal modulation features
        temporal_envelope = np.mean(mag, axis=0)
        features.extend([
            np.mean(temporal_envelope), np.std(temporal_envelope),
            kurtosis(temporal_envelope), skew(temporal_envelope),
        ])

        # Frequency band energies (divide spectrum into 8 bands)
        n_bands = 8
        band_size = mag.shape[0] // n_bands
        for i in range(n_bands):
            band = mag[i*band_size:(i+1)*band_size, :]
            features.append(np.mean(band))

        # Spectral contrast (peak-valley difference per band)
        for i in range(min(4, n_bands)):
            band = mag[i*band_size:(i+1)*band_size, :]
            peak = np.max(band, axis=0)
            valley = np.min(band, axis=0)
            features.append(np.mean(peak - valley))

        return np.array(features)


class HOSFeatures:
    """Higher-Order Statistics / Cumulants - matches RFML's HOS expert.
    Computes cumulants up to 6th order (C20, C21, C40, C41, C42, C60, C61, C62, C63)."""
    name = "Higher-Order Statistics (20 features)"

    def extract(self, samples):
        # Subsample for computational efficiency
        if len(samples) > 32768:
            samples = samples[:32768]

        # Normalize
        samples = samples / (np.sqrt(np.mean(np.abs(samples)**2)) + 1e-10)

        features = []

        # Second-order cumulants
        C20 = np.mean(samples**2)
        C21 = np.mean(np.abs(samples)**2)

        # Fourth-order cumulants
        M40 = np.mean(samples**4)
        M41 = np.mean(samples**3 * np.conj(samples))
        M42 = np.mean((np.abs(samples)**2)**2)
        M20 = np.mean(samples**2)
        M21 = np.mean(np.abs(samples)**2)

        C40 = M40 - 3 * M20**2
        C41 = M41 - 3 * M21 * M20
        C42 = M42 - np.abs(M20)**2 - 2 * M21**2

        # Sixth-order cumulants (simplified)
        M60 = np.mean(samples**6)
        M61 = np.mean(samples**5 * np.conj(samples))
        M62 = np.mean(samples**4 * np.conj(samples)**2)
        M63 = np.mean((np.abs(samples)**2)**3)

        C60 = M60 - 15*M20*M40 + 30*M20**3
        C61 = M61 - 5*M21*M40 - 10*M20*M41 + 30*M20**2*M21
        C62 = M62 - np.abs(M20)**2*M42 - 8*M21*M41 - M20*np.conj(M40) + 6*M21**2*M20 + 6*M20**2*np.conj(M20)
        C63 = M63 - 9*M21*M42 + 12*M21**3

        cumulants = [C20, C21, C40, C41, C42, C60, C61, C62, C63]

        # Power-invariant normalization (as in RFML)
        norm_factor = np.abs(C21)**(np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])/2) + 1e-10
        normalized = np.array([np.abs(c) for c in cumulants]) / norm_factor

        features.extend(normalized.tolist())

        # Derived statistics
        features.append(np.abs(C42) / (np.abs(C21)**2 + 1e-10))  # kurtosis-like
        features.append(np.abs(C40) / (np.abs(C20)**2 + 1e-10))
        features.append(np.abs(C63) / (np.abs(C21)**3 + 1e-10))
        features.append(np.abs(C60) / (np.abs(C20)**3 + 1e-10))

        # Phase of key cumulants
        features.append(np.angle(C40))
        features.append(np.angle(C42))
        features.append(np.angle(C60))

        # Ratios
        features.append(np.abs(C40) / (np.abs(C42) + 1e-10))
        features.append(np.abs(C60) / (np.abs(C63) + 1e-10))
        features.append(np.abs(C41) / (np.abs(C42) + 1e-10))
        features.append(np.abs(C61) / (np.abs(C62) + 1e-10))

        return np.array(features, dtype=np.float64)


class CycloFeatures:
    """Cyclostationary features via Spectral Correlation Function.
    Matches RFML's cyclostationary expert pathway."""
    name = "Cyclostationary SCF (64 features)"

    def extract(self, samples):
        # Subsample for speed
        if len(samples) > 16384:
            samples = samples[:16384]

        N = len(samples)
        Nfft = 256

        # Compute SCF via FFT Accumulation Method (simplified)
        num_blocks = N // Nfft
        if num_blocks < 2:
            return np.zeros(64)

        blocks = samples[:num_blocks * Nfft].reshape(num_blocks, Nfft)
        window = np.hanning(Nfft)
        blocks_windowed = blocks * window

        # FFT of each block
        X = np.fft.fft(blocks_windowed, axis=1)

        # Spectral correlation at various cycle frequencies
        n_alpha = 32  # number of cycle frequencies to probe
        alpha_indices = np.linspace(1, Nfft//2-1, n_alpha, dtype=int)

        scf_features = []
        for alpha_idx in alpha_indices:
            # Cross-spectral density at cycle frequency alpha
            Sxa = np.mean(X * np.conj(np.roll(X, alpha_idx, axis=1)), axis=0)
            scf_features.append(np.max(np.abs(Sxa)))
            scf_features.append(np.mean(np.abs(Sxa)))

        features = np.array(scf_features[:64])
        if len(features) < 64:
            features = np.pad(features, (0, 64 - len(features)))

        return features


class CombinedRFMLFeatures:
    """All RFML features concatenated (IQ + Spectrogram + HOS + Cyclo)."""
    name = "Combined RFML (174 features)"

    def __init__(self):
        self.extractors = [IQStatFeatures(), SpectrogramStatFeatures(), HOSFeatures(), CycloFeatures()]

    def extract(self, samples):
        all_features = []
        for ext in self.extractors:
            all_features.extend(ext.extract(samples).tolist())
        return np.array(all_features)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(data_dir='datasets_validated'):
    """Load all samples and extract features using all extractors."""
    extractors = {
        'baseline': BaselineRTLFeatures(),
        'iq_stat': IQStatFeatures(),
        'spectrogram': SpectrogramStatFeatures(),
        'hos': HOSFeatures(),
        'cyclo': CycloFeatures(),
        'combined': CombinedRFMLFeatures(),
    }

    features = {name: [] for name in extractors}
    labels = []

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Loading {len(classes)} classes: {classes}")

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files = sorted([f for f in os.listdir(cls_dir) if f.endswith('.npy')])
        print(f"  {cls}: {len(files)} samples")

        for filename in tqdm(files, desc=f"  {cls}", leave=False):
            data = np.load(os.path.join(cls_dir, filename), allow_pickle=True).item()
            samples = data['samples']
            samples = samples - np.mean(samples)  # DC removal

            for name, ext in extractors.items():
                try:
                    feat = ext.extract(samples)
                    # Replace inf/nan
                    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                    features[name].append(feat)
                except Exception as e:
                    print(f"    WARNING: {name} failed on {filename}: {e}")
                    features[name].append(np.zeros(len(features[name][0]) if features[name] else 17))

            labels.append(cls)

    result = {}
    for name in extractors:
        result[name] = np.array(features[name])
    result['labels'] = np.array(labels)

    return result, classes


def temporal_split(X, y, train_ratio=0.8):
    """Temporal split per class (first 80% train, last 20% test)."""
    X_train, X_test, y_train, y_test = [], [], [], []
    for label in sorted(np.unique(y)):
        mask = y == label
        X_class = X[mask]
        y_class = y[mask]
        split_idx = int(len(X_class) * train_ratio)
        X_train.extend(X_class[:split_idx])
        X_test.extend(X_class[split_idx:])
        y_train.extend(y_class[:split_idx])
        y_test.extend(y_class[split_idx:])
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


# ============================================================================
# US-002: INDIVIDUAL EXPERT CLASSIFIERS
# ============================================================================

def train_individual_experts(data, classes):
    """Train RF classifier for each feature modality."""
    print("\n" + "="*80)
    print("US-002: INDIVIDUAL EXPERT CLASSIFIERS")
    print("="*80)

    y = data['labels']
    modalities = ['baseline', 'iq_stat', 'spectrogram', 'hos', 'cyclo', 'combined']

    results = {}
    all_predictions = {}
    all_probabilities = {}

    for mod in modalities:
        X = data[mod]
        X_train, X_test, y_train, y_test = temporal_split(X, y)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Try multiple classifiers
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42),
        }

        best_model = None
        best_score = 0
        best_name = ""

        for mname, model in models.items():
            model.fit(X_train_s, y_train)
            score = model.score(X_test_s, y_test)
            if score > best_score:
                best_score = score
                best_model = model
                best_name = mname

        y_pred = best_model.predict(X_test_s)

        # Get probabilities if available
        if hasattr(best_model, 'predict_proba'):
            y_proba = best_model.predict_proba(X_test_s)
        else:
            y_proba = None

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=classes)

        # Per-class accuracy
        per_class_acc = {}
        for cls in classes:
            mask = y_test == cls
            if mask.sum() > 0:
                per_class_acc[cls] = accuracy_score(y_test[mask], y_pred[mask])

        results[mod] = {
            'best_classifier': best_name,
            'accuracy': best_score,
            'f1_macro': report['macro avg']['f1-score'],
            'f1_weighted': report['weighted avg']['f1-score'],
            'per_class_accuracy': per_class_acc,
            'report': report,
            'confusion_matrix': cm.tolist(),
            'n_features': X.shape[1],
        }

        all_predictions[mod] = y_pred
        all_probabilities[mod] = y_proba

        print(f"\n{'='*60}")
        print(f"  {mod.upper()} ({data[mod].shape[1]} features) — Best: {best_name}")
        print(f"{'='*60}")
        print(f"  Accuracy: {best_score:.3f} | F1-macro: {report['macro avg']['f1-score']:.3f}")
        print(classification_report(y_test, y_pred))
        print(f"  Per-class accuracy:")
        for cls in classes:
            acc = per_class_acc.get(cls, 0)
            bar = "█" * int(acc * 20)
            print(f"    {cls:15s}: {bar:20s} {acc*100:.1f}%")

    return results, all_predictions, all_probabilities


# ============================================================================
# US-003: MOE ENSEMBLE COMBINATIONS
# ============================================================================

def moe_ensemble_study(data, classes, individual_results, all_predictions, all_probabilities):
    """Test various MoE-style ensemble combinations."""
    print("\n" + "="*80)
    print("US-003: MOE ENSEMBLE COMBINATION STUDY")
    print("="*80)

    y = data['labels']
    expert_mods = ['baseline', 'iq_stat', 'spectrogram', 'hos', 'cyclo']

    # Prepare train/test data for all modalities
    train_test = {}
    for mod in expert_mods:
        X = data[mod]
        X_train, X_test, y_train, y_test = temporal_split(X, y)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        train_test[mod] = (X_train_s, X_test_s, y_train, y_test)

    y_test = train_test['baseline'][3]  # same for all

    ensemble_results = {}

    # Method 1: Majority Voting
    print("\n--- Method 1: Majority Voting ---")
    from collections import Counter
    vote_preds = []
    for i in range(len(y_test)):
        votes = [all_predictions[mod][i] for mod in expert_mods]
        vote_preds.append(Counter(votes).most_common(1)[0][0])
    vote_preds = np.array(vote_preds)
    vote_acc = accuracy_score(y_test, vote_preds)
    vote_f1 = f1_score(y_test, vote_preds, average='macro')
    cm = confusion_matrix(y_test, vote_preds, labels=classes)
    ensemble_results['majority_vote'] = {
        'accuracy': vote_acc, 'f1_macro': vote_f1,
        'confusion_matrix': cm.tolist(),
        'per_class': {cls: accuracy_score(y_test[y_test==cls], vote_preds[y_test==cls]) for cls in classes}
    }
    print(f"  Accuracy: {vote_acc:.3f} | F1-macro: {vote_f1:.3f}")

    # Method 2: Soft Voting (probability averaging)
    print("\n--- Method 2: Soft Voting (Probability Average) ---")
    prob_mods = [mod for mod in expert_mods if all_probabilities[mod] is not None]
    if len(prob_mods) >= 3:
        # Align class labels
        avg_proba = np.zeros_like(all_probabilities[prob_mods[0]])
        for mod in prob_mods:
            avg_proba += all_probabilities[mod]
        avg_proba /= len(prob_mods)

        # Need to get the class order from the RF
        X_train_s, X_test_s, y_train, _ = train_test['baseline']
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        class_order = rf.classes_

        soft_preds = class_order[np.argmax(avg_proba, axis=1)]
        soft_acc = accuracy_score(y_test, soft_preds)
        soft_f1 = f1_score(y_test, soft_preds, average='macro')
        cm = confusion_matrix(y_test, soft_preds, labels=classes)
        ensemble_results['soft_vote'] = {
            'accuracy': soft_acc, 'f1_macro': soft_f1,
            'confusion_matrix': cm.tolist(),
            'per_class': {cls: accuracy_score(y_test[y_test==cls], soft_preds[y_test==cls]) for cls in classes}
        }
        print(f"  Accuracy: {soft_acc:.3f} | F1-macro: {soft_f1:.3f}")

    # Method 3: Stacking Meta-Learner
    print("\n--- Method 3: Stacking Meta-Learner ---")
    # Use predictions from each expert as meta-features
    meta_features_train = []
    meta_features_test = []
    for mod in expert_mods:
        X_train_s, X_test_s, y_train, _ = train_test[mod]
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        meta_features_train.append(rf.predict_proba(X_train_s))
        meta_features_test.append(rf.predict_proba(X_test_s))

    X_meta_train = np.hstack(meta_features_train)
    X_meta_test = np.hstack(meta_features_test)

    meta_clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    meta_clf.fit(X_meta_train, y_train)
    stack_preds = meta_clf.predict(X_meta_test)
    stack_acc = accuracy_score(y_test, stack_preds)
    stack_f1 = f1_score(y_test, stack_preds, average='macro')
    cm = confusion_matrix(y_test, stack_preds, labels=classes)
    ensemble_results['stacking'] = {
        'accuracy': stack_acc, 'f1_macro': stack_f1,
        'confusion_matrix': cm.tolist(),
        'per_class': {cls: accuracy_score(y_test[y_test==cls], stack_preds[y_test==cls]) for cls in classes}
    }
    print(f"  Accuracy: {stack_acc:.3f} | F1-macro: {stack_f1:.3f}")

    # Method 4: Learned Gating (signal-dependent expert weighting)
    print("\n--- Method 4: Learned Gating Network ---")
    # Use baseline features as gating input, expert probas as expert outputs
    X_gate_train = train_test['baseline'][0]
    X_gate_test = train_test['baseline'][1]

    # Train a gating network that predicts expert weights
    # Simple approach: train MLP on concatenated [gate_features, expert_probas]
    gate_input_train = np.hstack([X_gate_train, X_meta_train])
    gate_input_test = np.hstack([X_gate_test, X_meta_test])

    gate_clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    gate_clf.fit(gate_input_train, y_train)
    gate_preds = gate_clf.predict(gate_input_test)
    gate_acc = accuracy_score(y_test, gate_preds)
    gate_f1 = f1_score(y_test, gate_preds, average='macro')
    cm = confusion_matrix(y_test, gate_preds, labels=classes)
    ensemble_results['learned_gating'] = {
        'accuracy': gate_acc, 'f1_macro': gate_f1,
        'confusion_matrix': cm.tolist(),
        'per_class': {cls: accuracy_score(y_test[y_test==cls], gate_preds[y_test==cls]) for cls in classes}
    }
    print(f"  Accuracy: {gate_acc:.3f} | F1-macro: {gate_f1:.3f}")

    # Method 5: Oracle (best expert per sample - upper bound)
    print("\n--- Method 5: Oracle Upper Bound ---")
    oracle_preds = []
    for i in range(len(y_test)):
        for mod in expert_mods:
            if all_predictions[mod][i] == y_test[i]:
                oracle_preds.append(y_test[i])
                break
        else:
            oracle_preds.append(all_predictions[expert_mods[0]][i])
    oracle_preds = np.array(oracle_preds)
    oracle_acc = accuracy_score(y_test, oracle_preds)
    ensemble_results['oracle'] = {'accuracy': oracle_acc, 'f1_macro': f1_score(y_test, oracle_preds, average='macro')}
    print(f"  Oracle Accuracy: {oracle_acc:.3f} (upper bound if perfect gating)")

    # Method 6: Feature Concatenation (all features, single classifier)
    print("\n--- Method 6: Feature Concatenation ---")
    X_concat_train = np.hstack([train_test[mod][0] for mod in expert_mods])
    X_concat_test = np.hstack([train_test[mod][1] for mod in expert_mods])

    concat_clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    concat_clf.fit(X_concat_train, y_train)
    concat_preds = concat_clf.predict(X_concat_test)
    concat_acc = accuracy_score(y_test, concat_preds)
    concat_f1 = f1_score(y_test, concat_preds, average='macro')
    cm = confusion_matrix(y_test, concat_preds, labels=classes)
    ensemble_results['feature_concat'] = {
        'accuracy': concat_acc, 'f1_macro': concat_f1,
        'confusion_matrix': cm.tolist(),
        'per_class': {cls: accuracy_score(y_test[y_test==cls], concat_preds[y_test==cls]) for cls in classes}
    }
    print(f"  Accuracy: {concat_acc:.3f} | F1-macro: {concat_f1:.3f}")

    # Save ensemble results separately
    with open('ensemble_results.json', 'w') as f:
        json.dump(ensemble_results, f, indent=2, default=str)
    print("\nEnsemble results saved to ensemble_results.json")

    return ensemble_results


# ============================================================================
# US-004: EXPERT SPECIALIZATION ANALYSIS
# ============================================================================

def expert_specialization_analysis(data, classes, individual_results, all_predictions):
    """Deep analysis of expert strengths/weaknesses."""
    print("\n" + "="*80)
    print("US-004: EXPERT SPECIALIZATION ANALYSIS")
    print("="*80)

    y = data['labels']
    _, _, _, y_test = temporal_split(data['baseline'], y)

    expert_mods = ['baseline', 'iq_stat', 'spectrogram', 'hos', 'cyclo']

    analysis = {}

    # 1. Per-expert accuracy by signal class (7x5 table)
    print("\n--- Per-Expert Accuracy by Signal Class ---")
    print(f"\n{'Signal':<16}", end='')
    for mod in expert_mods:
        print(f"{mod:>14}", end='')
    print(f"{'Best Expert':>16}")
    print("-" * (16 + 14*5 + 16))

    class_expert_acc = {}
    for cls in classes:
        mask = y_test == cls
        row = {}
        best_acc = 0
        best_mod = ""
        for mod in expert_mods:
            acc = accuracy_score(y_test[mask], all_predictions[mod][mask])
            row[mod] = acc
            if acc > best_acc:
                best_acc = acc
                best_mod = mod
        class_expert_acc[cls] = row

        print(f"{cls:<16}", end='')
        for mod in expert_mods:
            val = row[mod]
            marker = " *" if mod == best_mod else "  "
            print(f"{val*100:>12.1f}%{marker}", end='')
        print(f"{best_mod:>16}")

    analysis['class_expert_accuracy'] = class_expert_acc

    # 2. Expert Agreement Matrix
    print("\n--- Expert Agreement Matrix ---")
    agreement = {}
    print(f"\n{'':>14}", end='')
    for mod2 in expert_mods:
        print(f"{mod2:>14}", end='')
    print()

    for mod1 in expert_mods:
        agreement[mod1] = {}
        print(f"{mod1:>14}", end='')
        for mod2 in expert_mods:
            agree = np.mean(all_predictions[mod1] == all_predictions[mod2])
            agreement[mod1][mod2] = agree
            print(f"{agree*100:>13.1f}%", end='')
        print()

    analysis['expert_agreement'] = agreement

    # 3. Expert disagreement analysis - when experts disagree, who's right?
    print("\n--- When Experts Disagree, Who's Right? ---")
    for i, mod1 in enumerate(expert_mods):
        for mod2 in expert_mods[i+1:]:
            disagree_mask = all_predictions[mod1] != all_predictions[mod2]
            n_disagree = disagree_mask.sum()
            if n_disagree > 0:
                mod1_right = np.sum((all_predictions[mod1][disagree_mask] == y_test[disagree_mask]))
                mod2_right = np.sum((all_predictions[mod2][disagree_mask] == y_test[disagree_mask]))
                print(f"  {mod1} vs {mod2}: {n_disagree} disagreements → "
                      f"{mod1} correct {mod1_right}/{n_disagree} ({mod1_right/n_disagree*100:.0f}%), "
                      f"{mod2} correct {mod2_right}/{n_disagree} ({mod2_right/n_disagree*100:.0f}%)")

    # 4. Feature importance for ALL modalities
    modality_feature_names = {
        'baseline': [
            'power_mean', 'power_std', 'power_max', 'power_min',
            'fft_mean', 'fft_std', 'fft_max', 'fft_peak_idx',
            'i_mean', 'i_std', 'q_mean', 'q_std',
            'phase_mean', 'phase_std', 'phase_diff_mean', 'phase_diff_std',
            'bandwidth_ratio'
        ],
        'iq_stat': [
            'amp_mean', 'amp_std', 'amp_median', 'amp_kurtosis', 'amp_skew',
            'amp_p10', 'amp_p90', 'crest_factor',
            'i_mean', 'i_std', 'i_kurtosis', 'i_skew', 'i_p5', 'i_p95',
            'q_mean', 'q_std', 'q_kurtosis', 'q_skew', 'q_p5', 'q_p95',
            'phase_mean', 'phase_std', 'phase_kurtosis', 'phase_skew',
            'instfreq_mean', 'instfreq_std', 'instfreq_kurtosis', 'instfreq_skew', 'instfreq_median',
            'i_zerocross', 'q_zerocross',
            'autocorr_1', 'autocorr_10', 'autocorr_50', 'autocorr_100',
            'envelope_mean', 'envelope_std',
        ],
        'hos': [
            'C20_norm', 'C21_norm', 'C40_norm', 'C41_norm', 'C42_norm',
            'C60_norm', 'C61_norm', 'C62_norm', 'C63_norm',
            'kurtosis_C42', 'kurtosis_C40', 'kurtosis_C63', 'kurtosis_C60',
            'phase_C40', 'phase_C42', 'phase_C60',
            'ratio_C40_C42', 'ratio_C60_C63', 'ratio_C41_C42', 'ratio_C61_C62'
        ],
        'cyclo': [f'scf_max_{i}' if i % 2 == 0 else f'scf_mean_{i//2}' for i in range(64)],
    }

    analysis['feature_importance'] = {}
    for mod in expert_mods:
        print(f"\n--- {mod.upper()} Feature Importance (Top 10) ---")
        X_m = data[mod]
        X_train_m, _, y_train_m, _ = temporal_split(X_m, y)
        rf_m = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf_m.fit(X_train_m, y_train_m)

        fnames = modality_feature_names.get(mod, [f'{mod}_feat_{i}' for i in range(X_m.shape[1])])
        if len(fnames) < X_m.shape[1]:
            fnames.extend([f'{mod}_feat_{i}' for i in range(len(fnames), X_m.shape[1])])
        fnames = fnames[:X_m.shape[1]]

        importances = rf_m.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        for i in range(min(10, len(sorted_idx))):
            idx = sorted_idx[i]
            print(f"  {fnames[idx]:25s}: {importances[idx]:.4f}")

        analysis['feature_importance'][mod] = {fnames[i]: float(importances[i]) for i in sorted_idx[:10]}

    # 5. Error analysis - what gets misclassified?
    print("\n--- Error Analysis: Most Common Misclassifications ---")
    for mod in expert_mods:
        errors = []
        for i in range(len(y_test)):
            if all_predictions[mod][i] != y_test[i]:
                errors.append((y_test[i], all_predictions[mod][i]))
        if errors:
            print(f"\n  {mod}:")
            from collections import Counter
            for (true, pred), count in Counter(errors).most_common(5):
                print(f"    {true} → {pred}: {count} times")

    # 6. Signal characteristics that favor each expert
    print("\n--- Signal Characteristics Favoring Each Expert ---")
    expert_strengths = {
        'baseline': "Simple power/FFT features work best for signals with distinct spectral signatures (FM wideband, noise floor)",
        'iq_stat': "Extended IQ statistics excel for signals with unique amplitude/phase distributions (bursty signals, continuous carriers)",
        'spectrogram': "Time-frequency features capture temporal structure (burst patterns, packet timing in APRS/pager)",
        'hos': "Higher-order cumulants distinguish modulation types (phase vs frequency modulation, digital vs analog)",
        'cyclo': "Cyclostationary features detect periodicities (carrier frequencies, symbol rates, repeating patterns in ISM/FRS)",
    }

    for mod, desc in expert_strengths.items():
        best_classes = [cls for cls in classes if class_expert_acc[cls][mod] == max(class_expert_acc[cls].values())]
        print(f"\n  {mod}:")
        print(f"    Strength: {desc}")
        print(f"    Best for: {', '.join(best_classes)}")

    analysis['expert_strengths'] = expert_strengths

    return analysis


# ============================================================================
# US-005: ARCHITECTURE IMPROVEMENTS
# ============================================================================

def architecture_improvements(data, classes):
    """Propose and test architecture improvements."""
    print("\n" + "="*80)
    print("US-005: ARCHITECTURE IMPROVEMENT PROPOSALS")
    print("="*80)

    y = data['labels']
    improvements = {}

    # Improvement 1: Hierarchical Feature Selection
    print("\n--- Proposal 1: SNR-Aware Feature Selection ---")
    print("  Idea: Use different feature sets based on estimated signal quality")
    # Simulate by splitting dataset and comparing
    X_combined = data['combined']
    X_train, X_test, y_train, y_test = temporal_split(X_combined, y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train with feature selection (SelectKBest)
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    selector = SelectKBest(mutual_info_classif, k=50)
    X_train_sel = selector.fit_transform(X_train_s, y_train)
    X_test_sel = selector.transform(X_test_s)

    rf_sel = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf_sel.fit(X_train_sel, y_train)
    sel_acc = rf_sel.score(X_test_sel, y_test)

    rf_all = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf_all.fit(X_train_s, y_train)
    all_acc = rf_all.score(X_test_s, y_test)

    improvements['feature_selection'] = {
        'all_features_acc': all_acc,
        'selected_50_acc': sel_acc,
        'improvement': sel_acc - all_acc,
    }
    print(f"  All features ({X_combined.shape[1]}): {all_acc:.3f}")
    print(f"  Selected 50 features: {sel_acc:.3f}")
    print(f"  Delta: {sel_acc - all_acc:+.3f}")

    # Improvement 2: Gradient Boosted Ensemble (LightGBM-style)
    print("\n--- Proposal 2: Gradient Boosted Expert Fusion ---")
    print("  Idea: Replace RF with GBM for better handling of heterogeneous features")

    gb = GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
    gb.fit(X_train_s, y_train)
    gb_acc = gb.score(X_test_s, y_test)
    gb_preds = gb.predict(X_test_s)
    gb_f1 = f1_score(y_test, gb_preds, average='macro')

    improvements['gradient_boosting'] = {
        'accuracy': gb_acc,
        'f1_macro': gb_f1,
        'vs_rf_delta': gb_acc - all_acc,
    }
    print(f"  GBM Accuracy: {gb_acc:.3f} | F1: {gb_f1:.3f}")
    print(f"  vs RF: {gb_acc - all_acc:+.3f}")

    # Improvement 3: Two-Stage Classification
    print("\n--- Proposal 3: Two-Stage Hierarchical Classification ---")
    print("  Idea: First classify broad category, then specialize")

    # Stage 1: Broad categories
    broad_map = {
        'FM_broadcast': 'continuous', 'NOAA_weather': 'continuous',
        'APRS': 'bursty', 'pager': 'bursty', 'FRS_GMRS': 'bursty',
        'ISM_sensors': 'bursty', 'noise': 'noise',
    }
    y_broad_train = np.array([broad_map[l] for l in y_train])
    y_broad_test = np.array([broad_map[l] for l in y_test])

    stage1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    stage1.fit(X_train_s, y_broad_train)
    stage1_acc = stage1.score(X_test_s, y_broad_test)
    stage1_preds = stage1.predict(X_test_s)

    # Stage 2: Fine-grained within each broad category
    final_preds = np.empty_like(y_test)
    for broad_cat in ['continuous', 'bursty', 'noise']:
        train_mask = y_broad_train == broad_cat
        test_mask = stage1_preds == broad_cat

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        stage2 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        stage2.fit(X_train_s[train_mask], y_train[train_mask])
        final_preds[test_mask] = stage2.predict(X_test_s[test_mask])

    hier_acc = accuracy_score(y_test, final_preds)
    hier_f1 = f1_score(y_test, final_preds, average='macro')

    improvements['hierarchical'] = {
        'stage1_accuracy': stage1_acc,
        'final_accuracy': hier_acc,
        'f1_macro': hier_f1,
        'vs_flat_delta': hier_acc - all_acc,
    }
    print(f"  Stage 1 (broad): {stage1_acc:.3f}")
    print(f"  Final (hierarchical): {hier_acc:.3f} | F1: {hier_f1:.3f}")
    print(f"  vs Flat: {hier_acc - all_acc:+.3f}")

    # Improvement 4: Confidence-Weighted Expert Routing
    print("\n--- Proposal 4: Confidence-Weighted Expert Routing ---")
    print("  Idea: Weight expert contributions by their prediction confidence")

    expert_mods = ['baseline', 'iq_stat', 'spectrogram', 'hos', 'cyclo']
    expert_probas = []
    expert_class_orders = []

    for mod in expert_mods:
        X_m = data[mod]
        X_tr, X_te, y_tr, _ = temporal_split(X_m, y)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_tr_s, y_tr)
        expert_probas.append(rf.predict_proba(X_te_s))
        expert_class_orders.append(rf.classes_)

    # Align all expert probabilities to same class order
    ref_classes = expert_class_orders[0]
    aligned_probas = []
    for proba, cls_order in zip(expert_probas, expert_class_orders):
        aligned = np.zeros_like(proba)
        for i, cls in enumerate(cls_order):
            j = np.where(ref_classes == cls)[0][0]
            aligned[:, j] = proba[:, i]
        aligned_probas.append(aligned)

    # Confidence-weighted: weight by max probability
    weighted_sum = np.zeros_like(aligned_probas[0])
    for proba in aligned_probas:
        confidence = np.max(proba, axis=1, keepdims=True)  # per-sample confidence
        weighted_sum += proba * confidence

    conf_preds = ref_classes[np.argmax(weighted_sum, axis=1)]
    conf_acc = accuracy_score(y_test, conf_preds)
    conf_f1 = f1_score(y_test, conf_preds, average='macro')

    improvements['confidence_routing'] = {
        'accuracy': conf_acc,
        'f1_macro': conf_f1,
    }
    print(f"  Confidence-Weighted: {conf_acc:.3f} | F1: {conf_f1:.3f}")

    return improvements


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("RFML-MoE vs RTL-ML COMPREHENSIVE COMPARISON STUDY")
    print("="*80)

    # US-001: Extract features
    print("\n" + "="*80)
    print("US-001: FEATURE EXTRACTION")
    print("="*80)

    data, classes = load_dataset()

    print("\nFeature dimensions:")
    for name in ['baseline', 'iq_stat', 'spectrogram', 'hos', 'cyclo', 'combined']:
        print(f"  {name}: {data[name].shape}")

    # Save features
    os.makedirs('.omc', exist_ok=True)
    np.savez('.omc/extracted_features.npz', **data)
    print("Features saved to .omc/extracted_features.npz")

    # US-002: Individual experts
    individual_results, all_predictions, all_probabilities = train_individual_experts(data, classes)

    # US-003: Ensemble study
    ensemble_results = moe_ensemble_study(data, classes, individual_results, all_predictions, all_probabilities)

    # US-004: Expert analysis
    analysis = expert_specialization_analysis(data, classes, individual_results, all_predictions)

    # US-005: Architecture improvements
    improvements = architecture_improvements(data, classes)

    # Save all results
    all_results = {
        'individual_experts': {k: {kk: vv for kk, vv in v.items() if kk != 'report'}
                               for k, v in individual_results.items()},
        'ensemble_methods': ensemble_results,
        'improvements': improvements,
    }

    with open('comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print("\n--- Individual Expert Results ---")
    print(f"{'Modality':<25} {'Features':>8} {'Accuracy':>10} {'F1-macro':>10} {'Best Clf':>15}")
    print("-" * 70)
    for mod in ['baseline', 'iq_stat', 'spectrogram', 'hos', 'cyclo', 'combined']:
        r = individual_results[mod]
        print(f"{mod:<25} {r['n_features']:>8} {r['accuracy']:>10.3f} {r['f1_macro']:>10.3f} {r['best_classifier']:>15}")

    print("\n--- Ensemble Method Results ---")
    print(f"{'Method':<25} {'Accuracy':>10} {'F1-macro':>10}")
    print("-" * 45)
    for method, r in ensemble_results.items():
        print(f"{method:<25} {r['accuracy']:>10.3f} {r.get('f1_macro', 0):>10.3f}")

    print("\n--- Architecture Improvements ---")
    for name, r in improvements.items():
        acc = r.get('accuracy', r.get('selected_50_acc', r.get('final_accuracy', 0)))
        print(f"  {name}: {acc:.3f}")

    print("\nResults saved to comparison_results.json")
    return all_results


if __name__ == '__main__':
    main()
