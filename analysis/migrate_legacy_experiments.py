"""
기존 실험 결과를 새 추적 시스템으로 마이그레이션
"""

import json
from pathlib import Path
from experiment_tracker import ExperimentTracker, create_experiment_config

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "analysis" / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "analysis" / "experiments"


def migrate_cv_results():
    """CV 결과 마이그레이션"""
    cv_results_path = MODELS_DIR / "cv" / "cv_results.json"

    if not cv_results_path.exists():
        print(f"CV 결과 파일 없음: {cv_results_path}")
        return

    with open(cv_results_path, 'r', encoding='utf-8') as f:
        cv_results = json.load(f)

    tracker = ExperimentTracker(EXPERIMENTS_DIR)

    for model_type, results in cv_results.items():
        print(f"\n=== {model_type} 모델 마이그레이션 ===")

        # 설정 구성 (추정값)
        config = create_experiment_config(
            model_type=model_type,
            n_folds=results.get("n_folds", 5),
            epochs=100,
            batch_size=16,
            learning_rate=1e-3,
            random_state=42,
            target_shape=(128, 128),
            use_mel=False,
            fmax=6000
        )

        # 레거시 표시 추가
        config["legacy_import"] = True
        config["import_note"] = "기존 CV 실험에서 가져옴 - 데이터 분할 정보 없음"
        config["original_path"] = str(cv_results_path)

        # 실험 생성
        exp_id = tracker.create_experiment(f"cv_{model_type}", config)

        # 결과 저장
        tracker.save_final_results(results)

        print(f"  생성된 실험 ID: {exp_id}")
        print(f"  MAE: {results['overall_MAE_mean']:.4f} ± {results['overall_MAE_std']:.4f}")


def migrate_comparison_results():
    """모델 비교 결과 마이그레이션"""
    comparison_path = MODELS_DIR / "comparison_results.json"

    if not comparison_path.exists():
        print(f"비교 결과 파일 없음: {comparison_path}")
        return

    with open(comparison_path, 'r', encoding='utf-8') as f:
        comparison_results = json.load(f)

    tracker = ExperimentTracker(EXPERIMENTS_DIR)

    for model_type, results in comparison_results.items():
        print(f"\n=== {model_type} 모델 (단일 분할) 마이그레이션 ===")

        # 설정 구성 (추정값 - 4개 타겟 버전)
        config = create_experiment_config(
            model_type=model_type,
            n_folds=1,  # 단일 분할
            epochs=100,
            batch_size=16,
            learning_rate=1e-3,
            random_state=42,
            target_shape=(128, 128),
            use_mel=False,
            fmax=6000
        )

        config["legacy_import"] = True
        config["import_note"] = "기존 단일 분할 실험 - 4개 타겟 (audable_range_score 미포함)"
        config["original_path"] = str(comparison_path)
        config["target_columns"] = ["mid_freq_score", "irregularity", "low_high_freq", "regularity"]

        # 실험 생성
        exp_id = tracker.create_experiment(f"single_split_{model_type}", config)

        # 결과 저장
        tracker.save_final_results(results)

        print(f"  생성된 실험 ID: {exp_id}")
        print(f"  MAE: {results['overall_MAE']:.4f}")


def migrate_special_experiments():
    """특수 실험 결과 마이그레이션 (Mel, 분류기, 증강 등)"""
    tracker = ExperimentTracker(EXPERIMENTS_DIR)

    # 1. Mel Spectrogram 실험
    mel_path = MODELS_DIR / "mel_fmax4000" / "training_history.json"
    if mel_path.exists():
        print(f"\n=== Mel Spectrogram 실험 마이그레이션 ===")
        with open(mel_path, 'r', encoding='utf-8') as f:
            mel_history = json.load(f)

        config = create_experiment_config(
            model_type="simple",
            n_folds=1,
            epochs=100,
            batch_size=16,
            target_shape=(128, 128),
            use_mel=True,
            fmax=4000
        )
        config["legacy_import"] = True
        config["import_note"] = "Mel Spectrogram (fmax=4000) 실험"
        config["original_path"] = str(mel_path)

        exp_id = tracker.create_experiment("mel_fmax4000", config)

        # 최종 validation MAE 추출
        final_val_mae = mel_history.get("simple", {}).get("val_mae", [])[-1] if mel_history.get("simple", {}).get("val_mae") else None
        tracker.save_final_results({
            "training_history": mel_history,
            "final_val_mae": final_val_mae
        })
        print(f"  생성된 실험 ID: {exp_id}")

    # 2. 3-Class 분류기 실험
    classifier_path = MODELS_DIR / "classifier_3class" / "classification_results.json"
    if classifier_path.exists():
        print(f"\n=== 3-Class 분류기 실험 마이그레이션 ===")
        with open(classifier_path, 'r', encoding='utf-8') as f:
            classifier_results = json.load(f)

        config = create_experiment_config(
            model_type="classifier",
            n_folds=1,
            epochs=100,
            batch_size=16,
            target_shape=(128, 128),
            use_mel=False,
            fmax=6000
        )
        config["legacy_import"] = True
        config["import_note"] = "3-Class 분류 실험 (회귀 대신 분류)"
        config["original_path"] = str(classifier_path)
        config["task_type"] = "classification"
        config["num_classes"] = 3

        exp_id = tracker.create_experiment("classifier_3class", config)
        tracker.save_final_results(classifier_results)
        print(f"  생성된 실험 ID: {exp_id}")

    # 3. 데이터 증강 실험
    augmented_path = MODELS_DIR / "augmented_x2" / "augmented_results.json"
    if augmented_path.exists():
        print(f"\n=== 데이터 증강 (x2) 실험 마이그레이션 ===")
        with open(augmented_path, 'r', encoding='utf-8') as f:
            augmented_results = json.load(f)

        config = create_experiment_config(
            model_type="simple",
            n_folds=1,
            epochs=100,
            batch_size=16,
            target_shape=(128, 128),
            use_mel=False,
            fmax=6000
        )
        config["legacy_import"] = True
        config["import_note"] = "데이터 증강 (2배) 실험"
        config["original_path"] = str(augmented_path)
        config["augmentation"] = "x2"

        exp_id = tracker.create_experiment("augmented_x2", config)
        tracker.save_final_results(augmented_results)
        print(f"  생성된 실험 ID: {exp_id}")

    # 4. Spectral 특징 실험
    spectral_path = MODELS_DIR / "spectral" / "spectral_results.json"
    if spectral_path.exists():
        print(f"\n=== Spectral 특징 실험 마이그레이션 ===")
        with open(spectral_path, 'r', encoding='utf-8') as f:
            spectral_results = json.load(f)

        config = create_experiment_config(
            model_type="spectral",
            n_folds=1,
            epochs=100,
            batch_size=16,
            target_shape=(128, 128),
            use_mel=False,
            fmax=6000
        )
        config["legacy_import"] = True
        config["import_note"] = "Spectral 특징 기반 실험"
        config["original_path"] = str(spectral_path)

        exp_id = tracker.create_experiment("spectral_features", config)
        tracker.save_final_results(spectral_results)
        print(f"  생성된 실험 ID: {exp_id}")


def list_all_experiments():
    """모든 실험 목록 출력"""
    tracker = ExperimentTracker(EXPERIMENTS_DIR)
    experiments = tracker.list_experiments()

    print("\n" + "=" * 70)
    print("전체 실험 목록")
    print("=" * 70)

    if not experiments:
        print("  등록된 실험 없음")
        return

    print(f"{'ID':<45} | {'Name':<20} | Done")
    print("-" * 70)

    for exp in experiments:
        status = "O" if exp["completed"] else " "
        print(f"{exp['id']:<45} | {exp['name']:<20} | {status}")

    print(f"\n총 {len(experiments)}개 실험")


def main():
    print("=" * 70)
    print("레거시 실험 결과 마이그레이션")
    print("=" * 70)

    # 실험 디렉토리 생성
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # 마이그레이션 실행
    migrate_cv_results()
    migrate_comparison_results()
    migrate_special_experiments()

    # 결과 목록 출력
    list_all_experiments()

    print("\n" + "=" * 70)
    print("마이그레이션 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
