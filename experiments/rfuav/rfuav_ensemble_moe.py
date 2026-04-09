#!/usr/bin/env python3
"""
RFUAV Ensemble & MoE-Style Routing for 37-Class Drone RF Classification
=========================================================================
Combines statistical feature classifiers, raw IQ models, and DL spectrogram
models into ensemble/MoE pipelines. Mirrors rfml_comparison.py but scaled
to RFUAV (2623 train, 890 val, 37 classes).

Usage (on MI300X remote):
    sg render -c "python rfuav_ensemble_moe.py"

Prerequisite results (gracefully skipped if missing):
    - /home/rax/mtp/results/all_results.json          (DL spectrogram models)
    - /home/rax/mtp/results/rfuav_statistical_results.json  (stat feature models)
    - /home/rax/mtp/results/rfuav_raw_iq_results.json       (raw IQ models)

Saves: /home/rax/mtp/results/rfuav_ensemble_results.json
"""
import os
import sys
import json
import time
import warnings
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

MTP_DIR        = '/home/rax/mtp'
RESULT_DIR     = os.path.join(MTP_DIR, 'results')
SPEC_DIR       = os.path.join(MTP_DIR, 'spectrograms')
RAW_DIR        = os.path.join(MTP_DIR, 'raw')
MODEL_DIR      = os.path.join(MTP_DIR, 'models')
CHECKPOINT_DIR = MODEL_DIR

# Result files from prior pipelines
DL_RESULTS_FILE     = os.path.join(RESULT_DIR, 'all_results.json')
STAT_RESULTS_FILE   = os.path.join(RESULT_DIR, 'rfuav_statistical_results.json')
RAW_IQ_RESULTS_FILE = os.path.join(RESULT_DIR, 'rfuav_raw_iq_results.json')
ENSEMBLE_OUT_FILE   = os.path.join(RESULT_DIR, 'rfuav_ensemble_results.json')

os.makedirs(RESULT_DIR, exist_ok=True)

# DL models trained on RFUAV (from SESSION_CONTEXT Phase 5)
TIMM_MODELS = {
    'maxvit_base':        'maxvit_base_tf_224',
    'convnext_base':      'convnext_base',
    'efficientnetv2_l':   'tf_efficientnetv2_l',
    'convnext_large':     'convnext_large',
    'mobilenetv3_large':  'mobilenetv3_large_100',
    'vit_l_16':           'vit_large_patch16_224',
    'deit3_base':         'deit3_base_patch16_224',
    'swin_v2_base':       'swinv2_base_window12to16_192to256',
    'vit_b_32':           'vit_base_patch32_224',
    'eva02_base':         'eva02_base_patch14_224',
    'efficientnet_b0':    'efficientnet_b0',
    'resnet50':           'resnet50',
    'resnet18':           'resnet18',
}

IMG_SIZE = 224


# ============================================================================
# UTILITIES
# ============================================================================

def safe_json_load(path):
    """Load JSON file, return None if missing."""
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def accuracy_score_safe(y_true, y_pred):
    """Accuracy with zero-division safety."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def f1_macro_safe(y_true, y_pred):
    """Macro F1 with zero-division safety."""
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0))


def classification_report_dict(y_true, y_pred):
    """Full classification report as dict."""
    from sklearn.metrics import classification_report
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def per_class_accuracy(y_true, y_pred, classes):
    """Per-class accuracy dict."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    result = {}
    for cls in classes:
        mask = y_true == cls
        if mask.sum() > 0:
            result[cls] = float(np.mean(y_pred[mask] == y_true[mask]))
        else:
            result[cls] = 0.0
    return result


# ============================================================================
# PHASE 1: LOAD ALL MODEL PREDICTIONS
# ============================================================================

def load_dl_predictions():
    """
    Load DL spectrogram model predictions.

    Strategy:
      1. If saved prediction arrays exist (.npz cache), load directly.
      2. Else load checkpoints + run inference on val set.
      3. Fallback: load summary results (accuracy only, no per-sample preds).
    """
    print("\n" + "=" * 80)
    print("PHASE 1A: LOADING DL SPECTROGRAM MODEL PREDICTIONS")
    print("=" * 80)

    dl_preds = {}
    dl_probas = {}
    dl_meta = {}

    # --- Try loading pre-saved prediction arrays first ---
    pred_dir = os.path.join(RESULT_DIR, 'predictions')
    if os.path.isdir(pred_dir):
        for fname in sorted(os.listdir(pred_dir)):
            if fname.endswith('_preds.npz') and not fname.startswith(('stat_', 'iq_')):
                model_name = fname.replace('_preds.npz', '')
                data = np.load(os.path.join(pred_dir, fname), allow_pickle=True)
                dl_preds[model_name] = data['predictions']
                if 'probabilities' in data:
                    dl_probas[model_name] = data['probabilities']
                if 'labels' in data:
                    dl_meta['val_labels'] = data['labels']
                print(f"  [LOADED] {model_name} predictions from cache "
                      f"({len(data['predictions'])} samples)")

    if dl_preds:
        return dl_preds, dl_probas, dl_meta

    # --- Try running inference from checkpoints ---
    try:
        import torch
        import timm
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {DEVICE}")

        val_dir = os.path.join(SPEC_DIR, 'val')
        if not os.path.isdir(val_dir):
            print(f"  [SKIP] Val directory not found: {val_dir}")
            raise FileNotFoundError(val_dir)

        val_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                                num_workers=4, pin_memory=True)
        class_names = val_ds.classes
        num_classes = len(class_names)
        val_labels = [class_names[t] for _, t in val_ds.samples]
        dl_meta['val_labels'] = np.array(val_labels)
        dl_meta['class_names'] = class_names

        print(f"  Val set: {len(val_ds)} images, {num_classes} classes")

        # Find checkpoint files
        checkpoint_files = {}
        if os.path.isdir(CHECKPOINT_DIR):
            for fname in os.listdir(CHECKPOINT_DIR):
                if fname.endswith('_best.pt') or fname.endswith('_best.pth'):
                    model_key = fname.replace('_best.pt', '').replace('_best.pth', '')
                    checkpoint_files[model_key] = os.path.join(CHECKPOINT_DIR, fname)

        # Also check for YOLO checkpoint subdirectories
        yolo_dirs = []
        if os.path.isdir(CHECKPOINT_DIR):
            yolo_dirs = [d for d in os.listdir(CHECKPOINT_DIR)
                         if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))
                         and 'yolo' in d.lower()]

        # --- timm model inference ---
        for model_key, timm_name in TIMM_MODELS.items():
            ckpt_path = checkpoint_files.get(model_key)
            if ckpt_path is None:
                # Try alternate naming (underscores, hyphens)
                for cname, cpath in checkpoint_files.items():
                    if model_key.replace('_', '') in cname.replace('_', '').replace('-', ''):
                        ckpt_path = cpath
                        break
            if ckpt_path is None:
                print(f"  [SKIP] No checkpoint for {model_key}")
                continue

            print(f"  [INFER] {model_key} from {os.path.basename(ckpt_path)}...",
                  end=' ', flush=True)
            try:
                model = timm.create_model(timm_name, pretrained=False,
                                          num_classes=num_classes)
                state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
                if 'model_state_dict' in state:
                    model.load_state_dict(state['model_state_dict'])
                elif 'state_dict' in state:
                    model.load_state_dict(state['state_dict'])
                else:
                    model.load_state_dict(state)
                model = model.to(DEVICE)
                model.eval()

                all_preds_list = []
                all_probs_list = []
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(DEVICE)
                        with torch.amp.autocast(
                            device_type='cuda', dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()
                        ):
                            outputs = model(images)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()
                        preds = np.argmax(probs, axis=1)
                        all_preds_list.extend(preds)
                        all_probs_list.append(probs)

                pred_labels = np.array([class_names[p] for p in all_preds_list])
                prob_matrix = np.vstack(all_probs_list)

                dl_preds[model_key] = pred_labels
                dl_probas[model_key] = prob_matrix

                acc = accuracy_score_safe(val_labels, pred_labels)
                print(f"acc={acc:.4f}")

                # Cache predictions for future runs
                os.makedirs(pred_dir, exist_ok=True)
                np.savez(os.path.join(pred_dir, f'{model_key}_preds.npz'),
                         predictions=pred_labels, probabilities=prob_matrix,
                         labels=np.array(val_labels))

            except Exception as e:
                print(f"FAILED: {e}")

        # --- YOLO inference ---
        try:
            from ultralytics import YOLO
            for yolo_name in ['yolo11n-cls', 'yolo11s-cls', 'yolo8n-cls']:
                ckpt_candidates = [
                    os.path.join(CHECKPOINT_DIR, f'{yolo_name}_best.pt'),
                    os.path.join(CHECKPOINT_DIR,
                                 f'{yolo_name.replace("-", "_")}_best.pt'),
                ]
                for yd in yolo_dirs:
                    ckpt_candidates.append(
                        os.path.join(CHECKPOINT_DIR, yd, 'weights', 'best.pt'))

                ckpt_path = None
                for cp in ckpt_candidates:
                    if os.path.exists(cp):
                        ckpt_path = cp
                        break

                if ckpt_path is None:
                    continue

                print(f"  [INFER] {yolo_name} from {os.path.basename(ckpt_path)}...",
                      end=' ', flush=True)
                yolo_model = YOLO(ckpt_path)
                results = yolo_model.predict(val_dir, imgsz=IMG_SIZE,
                                             batch=64, verbose=False)

                yolo_preds = []
                yolo_probs_list = []
                for r in results:
                    pred_idx = r.probs.top1
                    yolo_preds.append(class_names[pred_idx])
                    yolo_probs_list.append(r.probs.data.cpu().numpy())

                dl_preds[yolo_name] = np.array(yolo_preds)
                dl_probas[yolo_name] = np.vstack(yolo_probs_list)

                acc = accuracy_score_safe(val_labels, yolo_preds)
                print(f"acc={acc:.4f}")

                os.makedirs(pred_dir, exist_ok=True)
                np.savez(os.path.join(pred_dir, f'{yolo_name}_preds.npz'),
                         predictions=np.array(yolo_preds),
                         probabilities=np.vstack(yolo_probs_list),
                         labels=np.array(val_labels))

        except ImportError:
            print("  [SKIP] ultralytics not installed, skipping YOLO models")
        except Exception as e:
            print(f"  [SKIP] YOLO inference failed: {e}")

    except ImportError as e:
        print(f"  [SKIP] PyTorch/timm not available: {e}")
    except FileNotFoundError:
        pass

    # --- Fallback: load summary results (no per-sample predictions) ---
    if not dl_preds:
        dl_summary = safe_json_load(DL_RESULTS_FILE)
        if dl_summary:
            print(f"\n  [INFO] Loaded DL summary results "
                  f"(accuracy only, no per-sample predictions)")
            for model_name, info in dl_summary.items():
                dl_meta[model_name] = {
                    'accuracy': info.get('accuracy',
                                         info.get('val_accuracy', 0)),
                    'f1_macro': info.get('f1_macro',
                                         info.get('val_f1', 0)),
                    'source': 'summary_only',
                }
            print(f"  Models in summary: {list(dl_summary.keys())}")

    print(f"\n  Total DL models with predictions: {len(dl_preds)}")
    return dl_preds, dl_probas, dl_meta


def load_statistical_predictions():
    """Load statistical feature model predictions."""
    print("\n" + "=" * 80)
    print("PHASE 1B: LOADING STATISTICAL FEATURE MODEL PREDICTIONS")
    print("=" * 80)

    stat_preds = {}
    stat_probas = {}
    stat_meta = {}

    results = safe_json_load(STAT_RESULTS_FILE)
    if results is None:
        print("  Statistical feature results not yet available.")
        print("  Run rfuav_statistical_features.py first.")
        return stat_preds, stat_probas, stat_meta

    # Load per-sample predictions if saved
    pred_dir = os.path.join(RESULT_DIR, 'predictions')
    for key in ['baseline', 'iq_stat', 'spectrogram_stat', 'hos', 'cyclo',
                'combined_rfml']:
        pred_file = os.path.join(pred_dir, f'stat_{key}_preds.npz')
        if os.path.exists(pred_file):
            data = np.load(pred_file, allow_pickle=True)
            stat_preds[f'stat_{key}'] = data['predictions']
            if 'probabilities' in data:
                stat_probas[f'stat_{key}'] = data['probabilities']
            if 'labels' in data:
                stat_meta['val_labels'] = data['labels']
            print(f"  [LOADED] stat_{key} ({len(data['predictions'])} samples)")

    # Store summary metrics
    for key, info in results.items():
        stat_meta[f'stat_{key}'] = {
            'accuracy': info.get('accuracy', 0),
            'f1_macro': info.get('f1_macro', 0),
            'best_classifier': info.get('best_classifier', 'unknown'),
            'n_features': info.get('n_features', 0),
        }

    print(f"  Statistical models with predictions: {len(stat_preds)}")
    print(f"  Statistical models in summary: "
          f"{len([k for k in stat_meta if k.startswith('stat_')])}")
    return stat_preds, stat_probas, stat_meta


def load_raw_iq_predictions():
    """Load raw IQ model predictions."""
    print("\n" + "=" * 80)
    print("PHASE 1C: LOADING RAW IQ MODEL PREDICTIONS")
    print("=" * 80)

    iq_preds = {}
    iq_probas = {}
    iq_meta = {}

    results = safe_json_load(RAW_IQ_RESULTS_FILE)
    if results is None:
        print("  Raw IQ results not yet available.")
        print("  Run rfuav_raw_iq_train.py first.")
        return iq_preds, iq_probas, iq_meta

    pred_dir = os.path.join(RESULT_DIR, 'predictions')
    for key in ['resnet1d', 'iq_cnn_transformer', 'inception_time',
                'se_resnet1d', 'cldnn', 'mcldnn', 'convnext_1d']:
        pred_file = os.path.join(pred_dir, f'iq_{key}_preds.npz')
        if os.path.exists(pred_file):
            data = np.load(pred_file, allow_pickle=True)
            iq_preds[f'iq_{key}'] = data['predictions']
            if 'probabilities' in data:
                iq_probas[f'iq_{key}'] = data['probabilities']
            if 'labels' in data:
                iq_meta['val_labels'] = data['labels']
            print(f"  [LOADED] iq_{key} ({len(data['predictions'])} samples)")

    for key, info in results.items():
        iq_meta[f'iq_{key}'] = {
            'accuracy': info.get('accuracy', 0),
            'f1_macro': info.get('f1_macro', 0),
        }

    print(f"  Raw IQ models with predictions: {len(iq_preds)}")
    return iq_preds, iq_probas, iq_meta


# ============================================================================
# PHASE 2: ENSEMBLE METHODS
# ============================================================================

def run_ensembles(all_preds, all_probas, val_labels, class_names):
    """
    Run all ensemble methods on available model predictions.

    Ensemble methods (matching rfml_comparison.py + MoE extensions):
      1. Majority Voting
      2. Soft Voting (probability averaging)
      3. Stacking Meta-Learner (LogisticRegression, 5-fold CV)
      4. Confidence-Weighted Routing
      5. Oracle (best model per sample -- upper bound)
      6. MoE Simple Gating (MLP on concatenated probabilities, 5-fold CV)
      7. Expert Choice Routing (per-modality-group confidence selection)
      8. Confidence Threshold Routing (best model >thresh, full ensemble else)
      9. Cross-Modality Voting (one vote per modality group)
    """
    print("\n" + "=" * 80)
    print("PHASE 2: ENSEMBLE METHODS")
    print("=" * 80)

    val_labels = np.array(val_labels)
    n_samples = len(val_labels)
    model_names = sorted(all_preds.keys())

    if len(model_names) < 2:
        print("  [SKIP] Need at least 2 models with predictions for ensembles.")
        return {}

    print(f"\n  Models available: {len(model_names)}")
    print(f"  Samples: {n_samples}")
    print(f"  Classes: {len(class_names)}")

    # Verify all prediction arrays have correct length
    valid_models = []
    for m in model_names:
        if len(all_preds[m]) == n_samples:
            valid_models.append(m)
        else:
            print(f"  [WARN] {m} has {len(all_preds[m])} predictions, "
                  f"expected {n_samples} -- skipping")
    model_names = valid_models

    # Categorize models by modality
    modality_groups = {
        'dl_spectrogram': [m for m in model_names
                           if not m.startswith('stat_')
                           and not m.startswith('iq_')],
        'statistical':    [m for m in model_names if m.startswith('stat_')],
        'raw_iq':         [m for m in model_names if m.startswith('iq_')],
    }
    for group, members in modality_groups.items():
        if members:
            print(f"  {group}: {members}")

    ensemble_results = {}

    # --- Individual model baselines ---
    print("\n--- Individual Model Baselines ---")
    individual_acc = {}
    for m in model_names:
        acc = accuracy_score_safe(val_labels, all_preds[m])
        f1 = f1_macro_safe(val_labels, all_preds[m])
        individual_acc[m] = acc
        print(f"  {m:35s}: acc={acc:.4f}  F1={f1:.4f}")

    best_individual = max(individual_acc, key=individual_acc.get)
    ensemble_results['individual_baselines'] = {
        m: {'accuracy': float(individual_acc[m]),
            'f1_macro': float(f1_macro_safe(val_labels, all_preds[m]))}
        for m in model_names
    }
    ensemble_results['best_individual'] = {
        'model': best_individual,
        'accuracy': float(individual_acc[best_individual]),
    }

    # ================================================================
    # METHOD 1: Majority Voting
    # ================================================================
    print("\n--- Method 1: Majority Voting ---")
    vote_preds = []
    for i in range(n_samples):
        votes = [all_preds[m][i] for m in model_names]
        winner = Counter(votes).most_common(1)[0][0]
        vote_preds.append(winner)
    vote_preds = np.array(vote_preds)
    vote_acc = accuracy_score_safe(val_labels, vote_preds)
    vote_f1 = f1_macro_safe(val_labels, vote_preds)
    ensemble_results['majority_vote'] = {
        'accuracy': float(vote_acc),
        'f1_macro': float(vote_f1),
        'per_class': per_class_accuracy(val_labels, vote_preds, class_names),
        'n_models': len(model_names),
    }
    print(f"  Accuracy: {vote_acc:.4f} | F1-macro: {vote_f1:.4f}")

    # ================================================================
    # METHOD 2: Soft Voting (probability averaging)
    # ================================================================
    print("\n--- Method 2: Soft Voting (Probability Average) ---")
    n_classes = len(class_names)
    prob_models = [m for m in model_names
                   if m in all_probas and all_probas[m] is not None
                   and all_probas[m].shape == (n_samples, n_classes)]

    if len(prob_models) >= 2:
        avg_proba = np.zeros((n_samples, n_classes))
        for m in prob_models:
            avg_proba += all_probas[m]
        avg_proba /= len(prob_models)

        soft_pred_idx = np.argmax(avg_proba, axis=1)
        soft_preds = np.array([class_names[i] for i in soft_pred_idx])
        soft_acc = accuracy_score_safe(val_labels, soft_preds)
        soft_f1 = f1_macro_safe(val_labels, soft_preds)
        ensemble_results['soft_vote'] = {
            'accuracy': float(soft_acc),
            'f1_macro': float(soft_f1),
            'per_class': per_class_accuracy(val_labels, soft_preds, class_names),
            'n_models': len(prob_models),
        }
        print(f"  Accuracy: {soft_acc:.4f} | F1-macro: {soft_f1:.4f} "
              f"({len(prob_models)} models)")
    else:
        print(f"  [SKIP] Only {len(prob_models)} models have probabilities "
              f"(need >= 2)")

    # ================================================================
    # METHOD 3: Stacking Meta-Learner (5-fold CV)
    # ================================================================
    print("\n--- Method 3: Stacking Meta-Learner ---")
    if len(prob_models) >= 2:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import LabelEncoder

        # Build meta-feature matrix from model probabilities
        meta_features = np.hstack([all_probas[m] for m in prob_models])
        n_meta_features = meta_features.shape[1]

        le = LabelEncoder()
        y_encoded = le.fit_transform(val_labels)

        # 5-fold CV to avoid overfitting the meta-learner on val set
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        stack_preds_cv = np.zeros(n_samples, dtype=int)

        for fold_idx, (train_idx, test_idx) in enumerate(
                skf.split(meta_features, y_encoded)):
            meta_lr = LogisticRegression(
                max_iter=2000, random_state=42, C=1.0,
                solver='lbfgs', multi_class='multinomial')
            meta_lr.fit(meta_features[train_idx], y_encoded[train_idx])
            stack_preds_cv[test_idx] = meta_lr.predict(meta_features[test_idx])

        stack_pred_labels = le.inverse_transform(stack_preds_cv)
        stack_acc = accuracy_score_safe(val_labels, stack_pred_labels)
        stack_f1 = f1_macro_safe(val_labels, stack_pred_labels)
        ensemble_results['stacking'] = {
            'accuracy': float(stack_acc),
            'f1_macro': float(stack_f1),
            'per_class': per_class_accuracy(
                val_labels, stack_pred_labels, class_names),
            'n_meta_features': n_meta_features,
            'method': '5-fold CV LogisticRegression on model probabilities',
        }
        print(f"  Accuracy: {stack_acc:.4f} | F1-macro: {stack_f1:.4f} "
              f"({n_meta_features} meta-features)")
    else:
        print("  [SKIP] Not enough probability models for stacking.")

    # ================================================================
    # METHOD 4: Confidence-Weighted Routing
    # ================================================================
    print("\n--- Method 4: Confidence-Weighted Routing ---")
    if len(prob_models) >= 2:
        weighted_sum = np.zeros((n_samples, n_classes))
        for m in prob_models:
            p = all_probas[m]
            # Weight each model's distribution by its own max confidence
            confidence = np.max(p, axis=1, keepdims=True)
            weighted_sum += p * confidence

        conf_pred_idx = np.argmax(weighted_sum, axis=1)
        conf_preds = np.array([class_names[i] for i in conf_pred_idx])
        conf_acc = accuracy_score_safe(val_labels, conf_preds)
        conf_f1 = f1_macro_safe(val_labels, conf_preds)
        ensemble_results['confidence_weighted'] = {
            'accuracy': float(conf_acc),
            'f1_macro': float(conf_f1),
            'per_class': per_class_accuracy(val_labels, conf_preds, class_names),
            'description': ('Weight each model probability by its '
                            'max confidence per sample'),
        }
        print(f"  Accuracy: {conf_acc:.4f} | F1-macro: {conf_f1:.4f}")
    else:
        print("  [SKIP] Not enough probability models.")

    # ================================================================
    # METHOD 5: Oracle (best model per sample -- upper bound)
    # ================================================================
    print("\n--- Method 5: Oracle Upper Bound ---")
    oracle_preds = []
    oracle_source = []
    for i in range(n_samples):
        correct_model = None
        for m in model_names:
            if all_preds[m][i] == val_labels[i]:
                correct_model = m
                break
        if correct_model is not None:
            oracle_preds.append(val_labels[i])
            oracle_source.append(correct_model)
        else:
            # No model got it right -- use the best overall model
            oracle_preds.append(all_preds[best_individual][i])
            oracle_source.append(f'{best_individual}(fallback)')
    oracle_preds = np.array(oracle_preds)
    oracle_acc = accuracy_score_safe(val_labels, oracle_preds)
    oracle_f1 = f1_macro_safe(val_labels, oracle_preds)

    oracle_counts = Counter(oracle_source)
    ensemble_results['oracle'] = {
        'accuracy': float(oracle_acc),
        'f1_macro': float(oracle_f1),
        'per_class': per_class_accuracy(val_labels, oracle_preds, class_names),
        'description': 'Perfect per-sample model selection (upper bound)',
        'model_contribution': {m: c for m, c in oracle_counts.most_common()},
    }
    print(f"  Oracle Accuracy: {oracle_acc:.4f} | F1-macro: {oracle_f1:.4f}")
    print(f"  Oracle model contributions (top 5):")
    for m, c in oracle_counts.most_common(5):
        print(f"    {m:35s}: {c:4d} samples ({c / n_samples * 100:.1f}%)")

    # ================================================================
    # METHOD 6: MoE Simple Gating (MLP, 5-fold CV)
    # ================================================================
    print("\n--- Method 6: MoE Simple Gating (MLP) ---")
    if len(prob_models) >= 2:
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_encoded = le.fit_transform(val_labels)

        gate_features = np.hstack([all_probas[m] for m in prob_models])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gate_preds_cv = np.zeros(n_samples, dtype=int)

        for fold_idx, (train_idx, test_idx) in enumerate(
                skf.split(gate_features, y_encoded)):
            gate_mlp = MLPClassifier(
                hidden_layer_sizes=(256, 128), max_iter=500,
                random_state=42, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=20)
            gate_mlp.fit(gate_features[train_idx], y_encoded[train_idx])
            gate_preds_cv[test_idx] = gate_mlp.predict(gate_features[test_idx])

        gate_pred_labels = le.inverse_transform(gate_preds_cv)
        gate_acc = accuracy_score_safe(val_labels, gate_pred_labels)
        gate_f1 = f1_macro_safe(val_labels, gate_pred_labels)
        ensemble_results['moe_simple_gating'] = {
            'accuracy': float(gate_acc),
            'f1_macro': float(gate_f1),
            'per_class': per_class_accuracy(
                val_labels, gate_pred_labels, class_names),
            'description': ('MLP gating network on concatenated model '
                            'probabilities (5-fold CV)'),
            'gate_input_dim': gate_features.shape[1],
        }
        print(f"  Accuracy: {gate_acc:.4f} | F1-macro: {gate_f1:.4f}")
    else:
        print("  [SKIP] Not enough probability models.")

    # ================================================================
    # METHOD 7: Expert Choice Routing (per modality group)
    # ================================================================
    print("\n--- Method 7: Expert Choice Routing (per modality group) ---")
    active_groups = {g: ms for g, ms in modality_groups.items() if ms}
    if len(active_groups) >= 2:
        group_preds = {}
        group_confidences = {}

        for group, members in active_groups.items():
            prob_members = [m for m in members if m in prob_models]
            if prob_members:
                # Soft vote within group
                group_proba = np.zeros((n_samples, n_classes))
                for m in prob_members:
                    group_proba += all_probas[m]
                group_proba /= len(prob_members)
                group_preds[group] = np.array(
                    [class_names[i] for i in np.argmax(group_proba, axis=1)])
                group_confidences[group] = np.max(group_proba, axis=1)
            else:
                # Majority vote fallback within group
                pred_members = [m for m in members if m in all_preds]
                if not pred_members:
                    continue
                gpreds = []
                gconf = []
                for i in range(n_samples):
                    votes = [all_preds[m][i] for m in pred_members]
                    winner, count = Counter(votes).most_common(1)[0]
                    gpreds.append(winner)
                    gconf.append(count / len(pred_members))
                group_preds[group] = np.array(gpreds)
                group_confidences[group] = np.array(gconf)

        if len(group_preds) >= 2:
            # Per-sample: pick the group with highest confidence
            expert_choice_preds = []
            expert_choice_source = []
            for i in range(n_samples):
                best_group = max(group_preds.keys(),
                                 key=lambda g: group_confidences[g][i])
                expert_choice_preds.append(group_preds[best_group][i])
                expert_choice_source.append(best_group)
            expert_choice_preds = np.array(expert_choice_preds)

            ec_acc = accuracy_score_safe(val_labels, expert_choice_preds)
            ec_f1 = f1_macro_safe(val_labels, expert_choice_preds)
            group_usage = Counter(expert_choice_source)
            ensemble_results['expert_choice_routing'] = {
                'accuracy': float(ec_acc),
                'f1_macro': float(ec_f1),
                'per_class': per_class_accuracy(
                    val_labels, expert_choice_preds, class_names),
                'description': ('Per-sample routing to highest-confidence '
                                'modality group'),
                'group_usage': {g: int(c) for g, c in group_usage.most_common()},
            }
            print(f"  Accuracy: {ec_acc:.4f} | F1-macro: {ec_f1:.4f}")
            print(f"  Group routing: {dict(group_usage.most_common())}")
        else:
            print(f"  [SKIP] Only {len(group_preds)} modality groups with "
                  f"predictions (need >= 2)")
    else:
        print(f"  [SKIP] Only {len(active_groups)} modality groups active "
              f"(need >= 2)")

    # ================================================================
    # METHOD 8: Confidence Threshold Routing
    # ================================================================
    print("\n--- Method 8: Confidence Threshold Routing ---")
    if (best_individual in all_probas
            and all_probas[best_individual] is not None
            and all_probas[best_individual].shape == (n_samples, n_classes)):
        best_proba = all_probas[best_individual]

        for threshold in [0.90, 0.95, 0.99]:
            best_conf = np.max(best_proba, axis=1)
            confident_mask = best_conf >= threshold
            n_confident = int(confident_mask.sum())

            ct_preds = np.empty(n_samples, dtype=object)

            # Confident samples: use best model directly
            if n_confident > 0:
                best_pred_idx = np.argmax(best_proba, axis=1)
                ct_preds[confident_mask] = np.array(
                    [class_names[i] for i in best_pred_idx[confident_mask]])

            # Uncertain samples: use full ensemble (soft vote)
            uncertain_mask = ~confident_mask
            n_uncertain = int(uncertain_mask.sum())
            if n_uncertain > 0 and len(prob_models) >= 2:
                ens_proba = np.zeros((n_samples, n_classes))
                for m in prob_models:
                    ens_proba += all_probas[m]
                ens_proba /= len(prob_models)
                uncertain_idx = np.argmax(
                    ens_proba[uncertain_mask], axis=1)
                ct_preds[uncertain_mask] = np.array(
                    [class_names[i] for i in uncertain_idx])
            elif n_uncertain > 0:
                ct_preds[uncertain_mask] = \
                    all_preds[best_individual][uncertain_mask]

            ct_acc = accuracy_score_safe(val_labels, ct_preds)
            ct_f1 = f1_macro_safe(val_labels, ct_preds)

            key = f'confidence_threshold_{int(threshold * 100)}'
            ensemble_results[key] = {
                'accuracy': float(ct_acc),
                'f1_macro': float(ct_f1),
                'per_class': per_class_accuracy(
                    val_labels, ct_preds, class_names),
                'threshold': threshold,
                'n_confident': n_confident,
                'n_uncertain': n_uncertain,
                'pct_confident': float(n_confident / n_samples * 100),
                'description': (
                    f'Best model ({best_individual}) for '
                    f'>{threshold * 100:.0f}% confident, '
                    f'full ensemble for uncertain'),
            }
            print(f"  Threshold {threshold:.0%}: acc={ct_acc:.4f} | "
                  f"F1={ct_f1:.4f} | "
                  f"confident={n_confident}/{n_samples} "
                  f"({n_confident / n_samples * 100:.1f}%)")
    else:
        print(f"  [SKIP] Best model ({best_individual}) has no probabilities.")

    # ================================================================
    # METHOD 9: Cross-Modality Voting (one vote per group)
    # ================================================================
    if len(active_groups) >= 2:
        print("\n--- Method 9: Cross-Modality Voting (1 vote per group) ---")
        group_votes = {}
        for group, members in active_groups.items():
            prob_members = [m for m in members
                           if m in all_probas
                           and all_probas[m] is not None
                           and all_probas[m].shape == (n_samples, n_classes)]
            if prob_members:
                gp = np.zeros((n_samples, n_classes))
                for m in prob_members:
                    gp += all_probas[m]
                gp /= len(prob_members)
                group_votes[group] = np.array(
                    [class_names[i] for i in np.argmax(gp, axis=1)])
            elif members:
                gpreds = []
                for i in range(n_samples):
                    votes = [all_preds[m][i] for m in members
                             if m in all_preds]
                    if votes:
                        gpreds.append(
                            Counter(votes).most_common(1)[0][0])
                    else:
                        gpreds.append(all_preds[best_individual][i])
                group_votes[group] = np.array(gpreds)

        if len(group_votes) >= 2:
            cross_mod_preds = []
            for i in range(n_samples):
                votes = [group_votes[g][i] for g in group_votes]
                cross_mod_preds.append(
                    Counter(votes).most_common(1)[0][0])
            cross_mod_preds = np.array(cross_mod_preds)
            cm_acc = accuracy_score_safe(val_labels, cross_mod_preds)
            cm_f1 = f1_macro_safe(val_labels, cross_mod_preds)
            ensemble_results['cross_modality_vote'] = {
                'accuracy': float(cm_acc),
                'f1_macro': float(cm_f1),
                'per_class': per_class_accuracy(
                    val_labels, cross_mod_preds, class_names),
                'description': 'One vote per modality group '
                               '(DL, Statistical, IQ)',
                'groups': list(group_votes.keys()),
            }
            print(f"  Accuracy: {cm_acc:.4f} | F1-macro: {cm_f1:.4f}")

    return ensemble_results


# ============================================================================
# PHASE 3: DIVERSITY & AGREEMENT ANALYSIS
# ============================================================================

def diversity_analysis(all_preds, all_probas, val_labels, class_names):
    """Analyze model diversity, agreement patterns, and complementarity."""
    print("\n" + "=" * 80)
    print("PHASE 3: DIVERSITY & AGREEMENT ANALYSIS")
    print("=" * 80)

    val_labels = np.array(val_labels)
    n_samples = len(val_labels)
    model_names = sorted(
        [m for m in all_preds if len(all_preds[m]) == n_samples])

    if len(model_names) < 2:
        print("  [SKIP] Need at least 2 models for diversity analysis.")
        return {}

    analysis = {}

    # --- 1. Pairwise Agreement Matrix ---
    print("\n--- Pairwise Agreement Matrix ---")
    agreement_matrix = {}
    for m1 in model_names:
        agreement_matrix[m1] = {}
        for m2 in model_names:
            agree = float(np.mean(all_preds[m1] == all_preds[m2]))
            agreement_matrix[m1][m2] = agree

    # Print compact version (first 8 models)
    show_models = model_names[:min(8, len(model_names))]
    abbrev = {m: m[:12] for m in show_models}
    header = f"{'':>13}" + "".join(
        f"{abbrev[m]:>13}" for m in show_models)
    print(header)
    for m1 in show_models:
        row = f"{abbrev[m1]:>13}"
        for m2 in show_models:
            row += f"{agreement_matrix[m1][m2] * 100:>12.1f}%"
        print(row)

    analysis['agreement_matrix'] = agreement_matrix

    # --- 2. Diversity metrics ---
    print("\n--- Ensemble Diversity Metrics ---")
    disagreements = []
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1:]:
            d = float(np.mean(all_preds[m1] != all_preds[m2]))
            disagreements.append(d)
    avg_disagreement = float(np.mean(disagreements))
    print(f"  Average pairwise disagreement: {avg_disagreement:.4f}")

    # Cohen's Kappa average
    kappas = []
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1:]:
            agree = np.mean(all_preds[m1] == all_preds[m2])
            classes_union = np.unique(
                np.concatenate([all_preds[m1], all_preds[m2]]))
            pe = sum(
                (np.mean(all_preds[m1] == c) * np.mean(all_preds[m2] == c))
                for c in classes_union)
            kappa = (agree - pe) / (1 - pe + 1e-10)
            kappas.append(float(kappa))
    avg_kappa = float(np.mean(kappas))
    print(f"  Average pairwise Kappa: {avg_kappa:.4f}")

    analysis['diversity'] = {
        'avg_pairwise_disagreement': avg_disagreement,
        'avg_pairwise_kappa': avg_kappa,
        'interpretation': (
            'Low kappa = high diversity = more ensemble benefit'
            if avg_kappa < 0.8 else
            'High kappa = low diversity = models agree, '
            'less ensemble benefit'),
    }

    # --- 3. Error correlation analysis ---
    print("\n--- Error Correlation "
          "(When model A errs, does B also err?) ---")
    error_masks = {}
    for m in model_names:
        error_masks[m] = (all_preds[m] != val_labels)

    error_corr = {}
    show_pairs = []
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1:]:
            both_wrong = float(
                np.mean(error_masks[m1] & error_masks[m2]))
            independent = float(
                np.mean(error_masks[m1]) * np.mean(error_masks[m2]))
            lift = both_wrong / (independent + 1e-10)
            error_corr[f'{m1}_vs_{m2}'] = {
                'both_wrong': both_wrong,
                'independent_baseline': independent,
                'lift': lift,
            }
            show_pairs.append((m1, m2, both_wrong, independent, lift))

    show_pairs.sort(key=lambda x: x[4], reverse=True)
    print(f"\n  Most correlated errors (make same mistakes):")
    for m1, m2, bw, ind, lift in show_pairs[:5]:
        print(f"    {m1[:20]:20s} vs {m2[:20]:20s}: lift={lift:.2f} "
              f"(both_wrong={bw:.4f}, independent={ind:.4f})")
    print(f"\n  Least correlated errors (complementary):")
    for m1, m2, bw, ind, lift in show_pairs[-5:]:
        print(f"    {m1[:20]:20s} vs {m2[:20]:20s}: lift={lift:.2f} "
              f"(both_wrong={bw:.4f}, independent={ind:.4f})")

    analysis['error_correlation'] = error_corr

    # --- 4. Per-class best expert ---
    print("\n--- Per-Class Best Expert ---")
    class_best = {}
    for cls in class_names:
        mask = val_labels == cls
        if mask.sum() == 0:
            continue
        best_model = None
        best_acc = -1
        for m in model_names:
            acc = float(np.mean(all_preds[m][mask] == val_labels[mask]))
            if acc > best_acc:
                best_acc = acc
                best_model = m
        class_best[cls] = {
            'best_model': best_model, 'accuracy': best_acc}
        n_cls = int(mask.sum())
        print(f"  {cls:25s} (n={n_cls:3d}): "
              f"{best_model:30s} acc={best_acc:.4f}")

    analysis['per_class_best_expert'] = class_best

    # --- 5. Modality group comparison ---
    print("\n--- Modality Group Comparison ---")
    groups = {
        'dl_spectrogram': [m for m in model_names
                           if not m.startswith('stat_')
                           and not m.startswith('iq_')],
        'statistical':    [m for m in model_names
                           if m.startswith('stat_')],
        'raw_iq':         [m for m in model_names
                           if m.startswith('iq_')],
    }
    group_accs = {}
    for group, members in groups.items():
        if not members:
            continue
        accs = [accuracy_score_safe(val_labels, all_preds[m])
                for m in members]
        group_accs[group] = {
            'n_models': len(members),
            'best_acc': float(max(accs)),
            'mean_acc': float(np.mean(accs)),
            'std_acc': float(np.std(accs)),
            'best_model': members[int(np.argmax(accs))],
        }
        print(f"  {group:20s}: {len(members)} models, "
              f"best={max(accs):.4f}, "
              f"mean={np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    analysis['modality_groups'] = group_accs

    # --- 6. Unique correct predictions (complementarity) ---
    print("\n--- Unique Correct Predictions (Complementarity) ---")
    for m in model_names:
        correct_m = (all_preds[m] == val_labels)
        others_correct = np.zeros(n_samples, dtype=bool)
        for m2 in model_names:
            if m2 != m:
                others_correct |= (all_preds[m2] == val_labels)
        unique_correct = correct_m & ~others_correct
        n_unique = int(unique_correct.sum())
        if n_unique > 0:
            print(f"  {m:35s}: {n_unique} unique correct samples")

    any_correct = np.zeros(n_samples, dtype=bool)
    for m in model_names:
        any_correct |= (all_preds[m] == val_labels)
    n_impossible = int((~any_correct).sum())
    print(f"\n  Samples where NO model is correct: "
          f"{n_impossible}/{n_samples} "
          f"({n_impossible / n_samples * 100:.1f}%)")
    analysis['n_impossible'] = n_impossible

    return analysis


# ============================================================================
# PHASE 4: HYPOTHESIS TESTING
# ============================================================================

def hypothesis_analysis(ensemble_results, dl_meta, stat_meta, iq_meta,
                        all_preds, val_labels, class_names):
    """Test key hypotheses about statistical features vs DL at RFUAV scale."""
    print("\n" + "=" * 80)
    print("PHASE 4: HYPOTHESIS TESTING")
    print("=" * 80)

    hypotheses = {}

    # ---------------------------------------------------------------
    # H1: Statistical features beat DL at RFUAV scale?
    # (RTL-ML: stat=100%, best DL=99.4% -> stat won)
    # (RFUAV: DL best=97.8% from all_results.json)
    # ---------------------------------------------------------------
    print("\n--- H1: Do statistical features beat DL at 2623+ samples? ---")
    print("  Context: On RTL-ML (800 samples), stat features (100%) > "
          "all DL (best 99.4%)")

    dl_models = {m: info for m, info in (dl_meta or {}).items()
                 if isinstance(info, dict) and 'accuracy' in info}
    stat_models = {m: info for m, info in (stat_meta or {}).items()
                   if isinstance(info, dict) and 'accuracy' in info}

    best_dl_acc = max(
        (info['accuracy'] for info in dl_models.values()), default=0)
    best_dl_name = max(
        dl_models, key=lambda m: dl_models[m]['accuracy'], default='N/A')
    best_stat_acc = max(
        (info['accuracy'] for info in stat_models.values()), default=0)
    best_stat_name = max(
        stat_models,
        key=lambda m: stat_models[m]['accuracy'], default='N/A')

    # Also check individual baselines from ensemble results
    if 'individual_baselines' in ensemble_results:
        for m, info in ensemble_results['individual_baselines'].items():
            if m.startswith('stat_') and info['accuracy'] > best_stat_acc:
                best_stat_acc = info['accuracy']
                best_stat_name = m
            elif (not m.startswith('stat_')
                  and not m.startswith('iq_')
                  and info['accuracy'] > best_dl_acc):
                best_dl_acc = info['accuracy']
                best_dl_name = m

    h1_result = 'inconclusive'
    if best_stat_acc > 0 and best_dl_acc > 0:
        gap = best_stat_acc - best_dl_acc
        if gap > 0.005:
            h1_result = 'stat_features_win'
        elif gap < -0.005:
            h1_result = 'dl_wins'
        else:
            h1_result = 'tie'
        print(f"  Best DL:   {best_dl_name} = {best_dl_acc:.4f}")
        print(f"  Best Stat: {best_stat_name} = {best_stat_acc:.4f}")
        print(f"  Gap: {gap:+.4f}")
        print(f"  Result: {h1_result.upper()}")
        confirmed = h1_result == 'stat_features_win'
        print(f"  {'CONFIRMED' if confirmed else 'REJECTED'}: "
              f"Statistical features "
              f"{'beat' if gap > 0 else 'lose to'} DL at RFUAV scale")
    else:
        print("  [INCONCLUSIVE] Missing predictions from one or "
              "both model types")

    hypotheses['h1_stat_beats_dl'] = {
        'question': 'Do statistical features beat DL at 2623+ samples?',
        'context': ('On RTL-ML (800 samples), stat features (100%) > '
                    'all DL (best 99.4%)'),
        'best_dl': {'model': best_dl_name, 'accuracy': float(best_dl_acc)},
        'best_stat': {
            'model': best_stat_name, 'accuracy': float(best_stat_acc)},
        'gap': (float(best_stat_acc - best_dl_acc)
                if best_stat_acc > 0 and best_dl_acc > 0 else None),
        'result': h1_result,
    }

    # ---------------------------------------------------------------
    # H2: Ensemble ceiling (oracle) vs best individual
    # ---------------------------------------------------------------
    print("\n--- H2: What is the ensemble accuracy ceiling (oracle)? ---")
    if 'oracle' in ensemble_results:
        oracle_acc = ensemble_results['oracle']['accuracy']
        best_ind = ensemble_results.get('best_individual', {})
        best_ind_acc = best_ind.get('accuracy', 0)
        headroom = oracle_acc - best_ind_acc
        print(f"  Oracle: {oracle_acc:.4f}")
        print(f"  Best individual: {best_ind.get('model', 'N/A')} "
              f"= {best_ind_acc:.4f}")
        print(f"  Headroom: {headroom:.4f} "
              f"({headroom * 100:.1f}% potential gain from perfect gating)")

        hypotheses['h2_oracle_ceiling'] = {
            'oracle_accuracy': float(oracle_acc),
            'best_individual_accuracy': float(best_ind_acc),
            'headroom': float(headroom),
            'interpretation': (
                'Large headroom -> models are complementary, '
                'ensemble can help'
                if headroom > 0.02 else
                'Small headroom -> models agree, '
                'ensemble adds little'),
        }
    else:
        print("  [SKIP] Oracle not computed.")

    # ---------------------------------------------------------------
    # H3: Which ensemble method is most practical?
    # ---------------------------------------------------------------
    print("\n--- H3: Best practical ensemble method ---")
    ensemble_accs = {}
    for method, info in ensemble_results.items():
        if (isinstance(info, dict) and 'accuracy' in info
                and method not in ('individual_baselines',
                                   'oracle', 'best_individual')):
            ensemble_accs[method] = info['accuracy']

    if ensemble_accs:
        best_method = max(ensemble_accs, key=ensemble_accs.get)
        print(f"  Best ensemble: {best_method} "
              f"= {ensemble_accs[best_method]:.4f}")
        print(f"\n  Ranking:")
        for method, acc in sorted(
                ensemble_accs.items(), key=lambda x: -x[1]):
            marker = " <<<" if method == best_method else ""
            print(f"    {method:35s}: {acc:.4f}{marker}")

        hypotheses['h3_best_ensemble'] = {
            'best_method': best_method,
            'accuracy': float(ensemble_accs[best_method]),
            'ranking': {m: float(a) for m, a in sorted(
                ensemble_accs.items(), key=lambda x: -x[1])},
        }
    else:
        print("  [SKIP] No ensemble results available.")

    # ---------------------------------------------------------------
    # H4: Does multi-modality help?
    # ---------------------------------------------------------------
    print("\n--- H4: Does multi-modality improve over single-modality? ---")
    best_single_modality = 0
    for m, info in ensemble_results.get('individual_baselines', {}).items():
        if info['accuracy'] > best_single_modality:
            best_single_modality = info['accuracy']

    best_ensemble = max(ensemble_accs.values()) if ensemble_accs else 0
    multi_mod_gain = best_ensemble - best_single_modality
    print(f"  Best single model: {best_single_modality:.4f}")
    print(f"  Best ensemble:     {best_ensemble:.4f}")
    print(f"  Gain: {multi_mod_gain:+.4f}")
    helps = multi_mod_gain > 0.001
    print(f"  Conclusion: Ensemble "
          f"{'helps' if helps else 'does NOT help'} at RFUAV scale")

    hypotheses['h4_multi_modality'] = {
        'best_single_model': float(best_single_modality),
        'best_ensemble': float(best_ensemble),
        'gain': float(multi_mod_gain),
        'helps': helps,
    }

    return hypotheses


# ============================================================================
# PHASE 5: FINAL REPORT
# ============================================================================

def print_final_report(ensemble_results, diversity, hypotheses,
                       dl_meta, stat_meta, iq_meta):
    """Print comprehensive comparison table."""
    print("\n" + "=" * 80)
    print("FINAL COMPARISON REPORT: RFUAV 37-Class Ensemble & MoE Study")
    print("=" * 80)

    # --- Summary table: Individual Models ---
    print("\n" + "-" * 80)
    print("INDIVIDUAL MODEL RESULTS")
    print("-" * 80)
    print(f"{'Model':35s} {'Type':15s} "
          f"{'Accuracy':>10s} {'F1-macro':>10s}")
    print("-" * 72)

    all_models = {}

    # DL models from meta
    for m, info in sorted(dl_meta.items()):
        if isinstance(info, dict) and 'accuracy' in info:
            all_models[m] = {
                'type': 'DL-Spec',
                'accuracy': info['accuracy'],
                'f1_macro': info.get('f1_macro', 0),
            }

    # Statistical models
    for m, info in sorted(stat_meta.items()):
        if isinstance(info, dict) and 'accuracy' in info:
            all_models[m] = {
                'type': 'Statistical',
                'accuracy': info['accuracy'],
                'f1_macro': info.get('f1_macro', 0),
            }

    # IQ models
    for m, info in sorted(iq_meta.items()):
        if isinstance(info, dict) and 'accuracy' in info:
            all_models[m] = {
                'type': 'Raw-IQ',
                'accuracy': info['accuracy'],
                'f1_macro': info.get('f1_macro', 0),
            }

    # Override with actual measured baselines if available
    if 'individual_baselines' in ensemble_results:
        for m, info in ensemble_results['individual_baselines'].items():
            mtype = 'DL-Spec'
            if m.startswith('stat_'):
                mtype = 'Statistical'
            elif m.startswith('iq_'):
                mtype = 'Raw-IQ'
            all_models[m] = {
                'type': mtype,
                'accuracy': info['accuracy'],
                'f1_macro': info['f1_macro'],
            }

    for m, info in sorted(
            all_models.items(), key=lambda x: -x[1]['accuracy']):
        print(f"  {m:33s} {info['type']:15s} "
              f"{info['accuracy']:>10.4f} {info['f1_macro']:>10.4f}")

    # --- Summary table: Ensemble Methods ---
    print("\n" + "-" * 80)
    print("ENSEMBLE METHOD RESULTS")
    print("-" * 80)
    print(f"{'Method':35s} {'Accuracy':>10s} "
          f"{'F1-macro':>10s} {'vs Best Ind.':>13s}")
    print("-" * 70)

    best_ind_acc = ensemble_results.get(
        'best_individual', {}).get('accuracy', 0)
    method_order = [
        'majority_vote', 'soft_vote', 'stacking',
        'confidence_weighted', 'moe_simple_gating',
        'expert_choice_routing', 'cross_modality_vote',
        'confidence_threshold_90', 'confidence_threshold_95',
        'confidence_threshold_99', 'oracle',
    ]
    for method in method_order:
        if method in ensemble_results:
            info = ensemble_results[method]
            acc = info['accuracy']
            f1 = info.get('f1_macro', 0)
            delta = acc - best_ind_acc
            marker = " *" if method == 'oracle' else ""
            print(f"  {method:33s} {acc:>10.4f} "
                  f"{f1:>10.4f} {delta:>+12.4f}{marker}")

    if best_ind_acc > 0:
        best_ind_name = ensemble_results.get(
            'best_individual', {}).get('model', 'N/A')
        print(f"\n  * Oracle is an upper bound, not achievable in practice.")
        print(f"  Best individual model: "
              f"{best_ind_name} ({best_ind_acc:.4f})")

    # --- Key hypothesis results ---
    print("\n" + "-" * 80)
    print("KEY FINDINGS")
    print("-" * 80)

    if 'h1_stat_beats_dl' in hypotheses:
        h1 = hypotheses['h1_stat_beats_dl']
        print(f"\n  H1 - Stat features vs DL at scale: "
              f"{h1['result'].upper()}")
        if h1['gap'] is not None:
            print(f"       Gap: {h1['gap']:+.4f}")

    if 'h2_oracle_ceiling' in hypotheses:
        h2 = hypotheses['h2_oracle_ceiling']
        print(f"\n  H2 - Oracle ceiling: {h2['oracle_accuracy']:.4f} "
              f"(headroom: {h2['headroom']:.4f})")

    if 'h3_best_ensemble' in hypotheses:
        h3 = hypotheses['h3_best_ensemble']
        print(f"\n  H3 - Best ensemble: "
              f"{h3['best_method']} ({h3['accuracy']:.4f})")

    if 'h4_multi_modality' in hypotheses:
        h4 = hypotheses['h4_multi_modality']
        verdict = 'HELPS' if h4['helps'] else 'NO BENEFIT'
        print(f"\n  H4 - Multi-modality: {verdict} "
              f"(gain: {h4['gain']:+.4f})")

    # --- Comparison with RTL-ML findings ---
    print("\n" + "-" * 80)
    print("COMPARISON WITH RTL-ML (800 samples, 7 classes)")
    print("-" * 80)

    # Fill in RFUAV values where available
    rfuav_dl = 'TBD'
    rfuav_stat = 'TBD'
    rfuav_winner = 'TBD'
    rfuav_mv = 'TBD'
    rfuav_cr = 'TBD'
    rfuav_oracle = 'TBD'

    if 'h1_stat_beats_dl' in hypotheses:
        h1 = hypotheses['h1_stat_beats_dl']
        if h1['best_dl']['accuracy'] > 0:
            rfuav_dl = f"{h1['best_dl']['accuracy'] * 100:.1f}%"
        if h1['best_stat']['accuracy'] > 0:
            rfuav_stat = f"{h1['best_stat']['accuracy'] * 100:.1f}%"
        if h1['result'] != 'inconclusive':
            rfuav_winner = ('Statistical' if h1['result'] == 'stat_features_win'
                            else 'DL' if h1['result'] == 'dl_wins'
                            else 'Tie')

    if 'majority_vote' in ensemble_results:
        rfuav_mv = f"{ensemble_results['majority_vote']['accuracy'] * 100:.1f}%"
    if 'confidence_weighted' in ensemble_results:
        rfuav_cr = f"{ensemble_results['confidence_weighted']['accuracy'] * 100:.1f}%"
    if 'oracle' in ensemble_results:
        rfuav_oracle = f"{ensemble_results['oracle']['accuracy'] * 100:.1f}%"

    print(f"""
  {'Metric':<35s} {'RTL-ML (800)':>15s} {'RFUAV (2623)':>15s}
  {'-' * 67}
  {'Best DL (spectrogram)':<35s} {'99.4%':>15s} {rfuav_dl:>15s}
  {'Best stat features':<35s} {'100.0%':>15s} {rfuav_stat:>15s}
  {'Winner':<35s} {'Statistical':>15s} {rfuav_winner:>15s}
  {'Majority Vote':<35s} {'100.0%':>15s} {rfuav_mv:>15s}
  {'Confidence Routing':<35s} {'100.0%':>15s} {rfuav_cr:>15s}
  {'Oracle':<35s} {'100.0%':>15s} {rfuav_oracle:>15s}
  {'Classes':<35s} {'7':>15s} {'37':>15s}
  {'Training samples':<35s} {'640':>15s} {'2623':>15s}
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("RFUAV ENSEMBLE & MoE ROUTING STUDY")
    print("37-class drone RF classification -- 2623 train, 890 val")
    print("=" * 80)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Result dir: {RESULT_DIR}")
    print(f"Spectrograms: {SPEC_DIR}")

    start_time = time.time()

    # Phase 1: Load all model predictions
    dl_preds, dl_probas, dl_meta = load_dl_predictions()
    stat_preds, stat_probas, stat_meta = load_statistical_predictions()
    iq_preds, iq_probas, iq_meta = load_raw_iq_predictions()

    # Merge all predictions into unified dicts
    all_preds = {}
    all_probas = {}
    all_preds.update(dl_preds)
    all_preds.update(stat_preds)
    all_preds.update(iq_preds)
    all_probas.update(dl_probas)
    all_probas.update(stat_probas)
    all_probas.update(iq_probas)

    # Determine val labels and class names
    val_labels = None
    class_names = None

    for source in [dl_meta, stat_meta, iq_meta]:
        if 'val_labels' in source:
            val_labels = source['val_labels']
        if 'class_names' in source:
            class_names = source['class_names']

    if val_labels is None and all_preds:
        # Infer from spectrogram val directory
        try:
            val_dir = os.path.join(SPEC_DIR, 'val')
            if os.path.isdir(val_dir):
                from torchvision import datasets
                val_ds = datasets.ImageFolder(val_dir)
                val_labels = np.array(
                    [val_ds.classes[t] for _, t in val_ds.samples])
                class_names = val_ds.classes
                print(f"\n  Inferred val labels from {val_dir}: "
                      f"{len(val_labels)} samples, "
                      f"{len(class_names)} classes")
        except Exception:
            pass

    if val_labels is None:
        print("\n" + "!" * 80)
        print("WARNING: No val labels available. "
              "Cannot run ensemble methods.")
        print("Run at least one model pipeline first "
              "to generate predictions.")
        print("!" * 80)

        # Save partial results (summary-only)
        summary = {
            'status': 'no_predictions_available',
            'dl_models': {
                m: info for m, info in dl_meta.items()
                if isinstance(info, dict) and 'accuracy' in info},
            'stat_models': {
                m: info for m, info in stat_meta.items()
                if isinstance(info, dict) and 'accuracy' in info},
            'iq_models': {
                m: info for m, info in iq_meta.items()
                if isinstance(info, dict) and 'accuracy' in info},
            'note': (
                'Run DL inference, rfuav_statistical_features.py, or '
                'rfuav_raw_iq_train.py to generate per-sample '
                'predictions, then re-run this script.'),
        }
        with open(ENSEMBLE_OUT_FILE, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nPartial results saved to {ENSEMBLE_OUT_FILE}")
        return summary

    if class_names is None:
        class_names = sorted(np.unique(val_labels).tolist())

    print(f"\n  Unified prediction pool: {len(all_preds)} models")
    print(f"  Val samples: {len(val_labels)}, "
          f"Classes: {len(class_names)}")

    # Phase 2: Run ensemble methods
    ensemble_results = run_ensembles(
        all_preds, all_probas, val_labels, class_names)

    # Phase 3: Diversity analysis
    diversity = diversity_analysis(
        all_preds, all_probas, val_labels, class_names)

    # Phase 4: Hypothesis testing
    hypotheses = hypothesis_analysis(
        ensemble_results, dl_meta, stat_meta, iq_meta,
        all_preds, val_labels, class_names)

    # Phase 5: Final report
    print_final_report(
        ensemble_results, diversity, hypotheses,
        dl_meta, stat_meta, iq_meta)

    # Save comprehensive results
    elapsed = time.time() - start_time

    final_results = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_models': len(all_preds),
            'n_val_samples': int(len(val_labels)),
            'n_classes': len(class_names),
            'class_names': list(class_names),
            'elapsed_seconds': round(elapsed, 1),
        },
        'ensemble_results': ensemble_results,
        'diversity_analysis': {
            k: v for k, v in diversity.items()
            if k != 'agreement_matrix'  # Too large for JSON
        },
        'hypotheses': hypotheses,
        'modality_groups': diversity.get('modality_groups', {}),
    }

    with open(ENSEMBLE_OUT_FILE, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {ENSEMBLE_OUT_FILE}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 80}")

    return final_results


if __name__ == '__main__':
    main()
