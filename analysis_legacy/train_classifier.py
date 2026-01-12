"""
분류 모델 학습 스크립트
역할: 연속 점수를 등급으로 변환하여 분류 문제로 학습
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU 사용 가능: {len(gpus)}개")
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from data_loader import AudioDataLoader


def score_to_class(scores, num_classes=3):
    """
    연속 점수를 등급으로 변환

    매개변수:
        scores: (N, 4) 형태의 점수 배열
        num_classes: 등급 수 (3: Good/Normal/Poor)

    반환값:
        overall_class: (N,) 형태의 등급 레이블
    """
    # 4개 점수의 평균을 전체 점수로 사용
    overall_score = scores.mean(axis=1)

    # 점수 범위: 1~5
    # 3등급: Poor(1-2.33), Normal(2.33-3.67), Good(3.67-5)
    if num_classes == 3:
        bins = [0, 2.33, 3.67, 6]
        labels = [0, 1, 2]  # Poor, Normal, Good
    elif num_classes == 5:
        bins = [0, 1.8, 2.6, 3.4, 4.2, 6]
        labels = [0, 1, 2, 3, 4]  # 1~5등급
    else:
        raise ValueError(f"지원하지 않는 등급 수: {num_classes}")

    classes = np.digitize(overall_score, bins[1:-1])
    return classes


def build_classifier(input_shape=(128, 128, 1), num_classes=3):
    """
    분류용 CNN 모델

    매개변수:
        input_shape: 입력 형태
        num_classes: 출력 클래스 수

    반환값:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape, name='input')

    # Conv Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    # Conv Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    # Conv Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    # Conv Block 4
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output (Classification)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='AudioClassifier')

    return model


def create_callbacks(model_name, save_dir):
    """학습 콜백 생성"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / f"{model_name}_best.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    return callbacks


def evaluate_classifier(model, X_test, y_test, class_names, num_classes):
    """
    분류 모델 평가

    매개변수:
        model: 학습된 모델
        X_test: 테스트 입력
        y_test: 테스트 타겟 (원-핫 인코딩)
        class_names: 클래스 이름 리스트
        num_classes: 전체 클래스 수

    반환값:
        평가 결과 딕셔너리
    """
    # 예측
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 정확도
    accuracy = accuracy_score(y_true, y_pred)

    # 분류 리포트 (labels 지정으로 모든 클래스 포함)
    labels = list(range(num_classes))
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # 혼동 행렬
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

    return results


def print_results(results, class_names):
    """평가 결과 출력"""
    print(f"\n{'='*60}")
    print("분류 모델 평가 결과")
    print(f"{'='*60}")
    print(f"정확도: {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")

    print(f"\n클래스별 성능:")
    for name in class_names:
        if name in results['classification_report']:
            metrics = results['classification_report'][name]
            print(f"  {name:10s} - Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}")

    print(f"\n혼동 행렬:")
    cm = np.array(results['confusion_matrix'])
    header = "예측→  " + "  ".join([f"{name[:4]:>5s}" for name in class_names])
    print(f"  {header}")
    for i, name in enumerate(class_names):
        row = "  ".join([f"{v:>5d}" for v in cm[i]])
        print(f"  {name[:4]:5s}  {row}")


def main():
    parser = argparse.ArgumentParser(description='EGAI 분류 모델 학습')
    parser.add_argument('--num_classes', type=int, default=3,
                        choices=[3, 5], help='등급 수 (3 또는 5)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='에포크 수')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='배치 크기')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='데이터 디렉토리')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='모델 저장 디렉토리')

    args = parser.parse_args()

    # 경로 설정
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    save_dir = Path(args.save_dir) if args.save_dir else PROJECT_ROOT / "analysis" / "models" / f"classifier_{args.num_classes}class"

    # 클래스 이름
    if args.num_classes == 3:
        class_names = ['Poor', 'Normal', 'Good']
    else:
        class_names = ['1등급', '2등급', '3등급', '4등급', '5등급']

    print("=" * 60)
    print("EGAI 분류 모델 학습")
    print("=" * 60)
    print(f"등급 수: {args.num_classes}")
    print(f"등급 이름: {class_names}")
    print(f"에포크: {args.epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"데이터 경로: {data_dir}")
    print(f"저장 경로: {save_dir}")

    # 데이터 로더 초기화 (STFT 사용)
    loader = AudioDataLoader(
        data_dir=data_dir,
        target_shape=(128, 128),
        use_mel=False,
        fmax=6000
    )

    # 메타데이터 로드
    loader.load_metadata()

    # 데이터 분할
    train_idx, val_idx, test_idx = loader.get_splits()

    # 데이터 준비
    print("\n데이터 준비 중...")
    print("  Train 데이터 로드...")
    X_train, y_train_reg = loader.prepare_dataset(train_idx, model_type='simple')
    print("  Val 데이터 로드...")
    X_val, y_val_reg = loader.prepare_dataset(val_idx, model_type='simple')
    print("  Test 데이터 로드...")
    X_test, y_test_reg = loader.prepare_dataset(test_idx, model_type='simple')

    # 회귀 타겟 → 분류 타겟 변환
    y_train_class = score_to_class(y_train_reg, args.num_classes)
    y_val_class = score_to_class(y_val_reg, args.num_classes)
    y_test_class = score_to_class(y_test_reg, args.num_classes)

    # 원-핫 인코딩
    y_train = keras.utils.to_categorical(y_train_class, args.num_classes)
    y_val = keras.utils.to_categorical(y_val_class, args.num_classes)
    y_test = keras.utils.to_categorical(y_test_class, args.num_classes)

    # 클래스 분포 출력
    print(f"\n클래스 분포:")
    for i, name in enumerate(class_names):
        train_count = (y_train_class == i).sum()
        val_count = (y_val_class == i).sum()
        test_count = (y_test_class == i).sum()
        print(f"  {name}: Train={train_count}, Val={val_count}, Test={test_count}")

    print(f"\n입력 형태: {X_train.shape}")
    print(f"타겟 형태: {y_train.shape}")

    # 모델 생성
    print("\n모델 생성...")
    model = build_classifier(num_classes=args.num_classes)
    print(f"총 파라미터: {model.count_params():,}")

    # 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 콜백
    callbacks = create_callbacks('classifier', save_dir)

    # 학습
    print(f"\n학습 시작 (epochs={args.epochs}, batch_size={args.batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 평가
    results = evaluate_classifier(model, X_test, y_test, class_names, args.num_classes)
    print_results(results, class_names)

    # 모델 저장
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir / "classifier_final.keras"))
    print(f"\n모델 저장: {save_dir}/classifier_final.keras")

    # 결과 저장
    results_path = save_dir / "classification_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        serializable_results = {
            'accuracy': float(results['accuracy']),
            'confusion_matrix': results['confusion_matrix'],
            'class_names': class_names
        }
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {results_path}")

    print("\n" + "=" * 60)
    print("분류 학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
