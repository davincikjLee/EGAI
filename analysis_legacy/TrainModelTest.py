"""
🧪 학습된 모델 테스트 스크립트 (TrainModelTest.py)
역할:
- 1201번부터 끝까지 데이터를 테스트 데이터로 사용
- 학습된 모델 로드 및 성능 평가
- Confusion Matrix 및 상세 분석
"""

# ========== 필요한 라이브러리 임포트 ==========
import os  # 파일 경로 처리
import gc  # 가비지 컬렉터 (메모리 정리)
import numpy as np  # 수치 연산
import pandas as pd  # CSV 파일 처리
import tensorflow as tf  # 딥러닝 프레임워크
from tensorflow import keras  # Keras API
from sklearn.preprocessing import LabelEncoder  # 레이블 인코딩
from sklearn.metrics import confusion_matrix, classification_report  # 평가 지표
import matplotlib.pyplot as plt  # 그래프 그리기
import seaborn as sns  # 히트맵 시각화
import json  # JSON 파일 처리

# 우리가 만든 모듈
from audio_preprocessing import AudioPreprocessor  # 음성 전처리
from fault_diagnosis_model import FaultDiagnosisModel  # 모델 구조
from train import FaultDiagnosisTrainer  # 트레이너 클래스 (prepare_data_from_paths 사용)

# ========== GPU 메모리 최적화 설정 ==========
print("=" * 60)
print("🔍 GPU 환경 설정")
print("=" * 60)

# TensorFlow 세션 클리어 (기존 메모리 정리)
tf.keras.backend.clear_session()

# 가비지 컬렉션 실행 (메모리 회수)
gc.collect()

# GPU 디바이스 목록 가져오기
gpus = tf.config.list_physical_devices('GPU')

if gpus:  # GPU가 감지된 경우
    try:
        # 동적 메모리 할당 활성화 (필요한 만큼만 사용)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("✅ GPU 메모리 최적화 완료")
        print(f"   - 감지된 GPU: {len(gpus)}개")
        print(f"   - 동적 할당: 활성화 (필요한 만큼 사용)")

    except RuntimeError as e:  # 설정 실패 시
        print(f"⚠️ GPU 설정 실패: {e}")
else:  # GPU 미감지 시
    print("❌ GPU를 찾을 수 없음, CPU로 테스트를 진행합니다.")

print("=" * 60)
print()


# ========== GPU 메모리 확인 함수 ==========
def check_gpu_memory():
    """GPU 메모리 사용량 확인"""
    import subprocess  # 외부 명령 실행
    try:
        # nvidia-smi 명령으로 GPU 메모리 정보 가져오기
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'],  # CSV 형식으로 출력
            capture_output=True,  # 출력 캡처
            text=True  # 텍스트로 받기
        )

        if result.returncode == 0:  # 명령 성공 시
            # 사용량과 전체 메모리 파싱
            used, total = map(int, result.stdout.strip().split(','))
            # 출력
            print(f"🖥️ GPU 메모리: {used}/{total} MB ({used / total * 100:.1f}%)")
        else:  # 명령 실패 시
            print("⚠️ GPU 메모리 정보를 가져올 수 없음")

    except Exception as e:  # nvidia-smi 실행 불가 시
        print(f"⚠️ nvidia-smi 실행 실패: {e}")


# ========== 메인 실행 부분 ==========
if __name__ == "__main__":

    # 시작 메시지
    print("=" * 60)
    print("🧪 학습된 모델 테스트 시작")
    print("=" * 60)

    # ========== 1. 학습된 모델 정보 로드 ==========
    print("\n📂 Step 1: 학습된 모델 정보 로드")
    print("-" * 60)

    # 모델 정보 JSON 파일 로드
    model_info_path = 'model_info_v1.json'  # 학습 시 저장한 파일

    # 파일 존재 확인
    if not os.path.exists(model_info_path):
        print(f"❌ 오류: {model_info_path} 파일을 찾을 수 없습니다!")
        print("먼저 train.py를 실행하여 모델을 학습시켜주세요.")
        exit(1)  # 프로그램 종료

    # JSON 파일 읽기
    with open(model_info_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)  # 딕셔너리로 로드

    # 모델 정보 출력
    print("✅ 모델 정보 로드 완료!")
    print(f"   - 버전: {model_info['version']}")
    print(f"   - 클래스 수: {model_info['num_classes']}")
    print(f"   - 학습 정확도: {model_info['train_acc']:.4f}")
    print(f"   - 검증 정확도: {model_info['val_acc']:.4f}")
    print(f"   - 입력 크기: {model_info['target_shape']}")

    # ========== 2. 레이블 매핑 로드 ==========
    print("\n📋 Step 2: 레이블 매핑 로드")
    print("-" * 60)

    # 레이블 매핑 JSON 파일 로드
    label_mapping_path = 'label_mapping.json'

    # 파일 존재 확인
    if not os.path.exists(label_mapping_path):
        print(f"❌ 오류: {label_mapping_path} 파일을 찾을 수 없습니다!")
        exit(1)

    # JSON 파일 읽기
    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)

    # 레이블 정보 출력
    print("✅ 레이블 매핑 로드 완료!")
    print(f"   - 레이블 컬럼: {label_mapping['label_column']}")
    print(f"   - 클래스 개수: {label_mapping['num_classes']}")
    print(f"\n클래스 목록:")

    # 클래스 이름과 인코딩 번호 출력
    for original, encoded in list(label_mapping['encoding'].items())[:10]:
        count = label_mapping['class_distribution'].get(original, 0)
        print(f"   {original} → {encoded} (학습 데이터: {count}개)")

    # 클래스가 10개 이상이면 '...' 표시
    if label_mapping['num_classes'] > 10:
        print(f"   ... (총 {label_mapping['num_classes']}개)")

    # ========== 3. 학습된 모델 로드 ==========
    print("\n🧠 Step 3: 학습된 모델 로드")
    print("-" * 60)

    # SavedModel 형식 경로 (폴더)
    model_path_savedmodel = 'best_fault_diagnosis_model_v1/'

    # SavedModel 형식 시도 (우선)
    if os.path.exists(model_path_savedmodel):
        try:
            # SavedModel 형식으로 로드
            model = keras.models.load_model(model_path_savedmodel)
            print(f"✅ 모델 로드 완료: {model_path_savedmodel} (SavedModel)")
        except Exception as e:
            print(f"❌ SavedModel 로드 실패: {e}")
            print("가중치로 모델 재구성 시도...")

            # 가중치로 재구성
            model = FaultDiagnosisModel(num_classes=label_mapping['num_classes'])

            # 더미 입력으로 모델 빌드 (가중치 로드 전 필수)
            dummy_input = [
                tf.zeros((1, 128, 128, 1)),  # Full spectrogram
                tf.zeros((1, 128, 128, 1))  # Percussive spectrogram
            ]
            _ = model(dummy_input)  # 모델 구조 초기화

            # 가중치 로드
            model.load_weights('fault_diagnosis_model.h5')
            print("✅ 가중치로 모델 재구성 완료")

    else:
        # SavedModel이 없으면 가중치로만 로드
        print("ℹ️ SavedModel 없음, 가중치로 모델 재구성...")

        # 모델 인스턴스 생성
        model = FaultDiagnosisModel(num_classes=label_mapping['num_classes'])

        # 더미 입력으로 모델 빌드
        dummy_input = [
            tf.zeros((1, 128, 128, 1)),
            tf.zeros((1, 128, 128, 1))
        ]
        _ = model(dummy_input)

        # 가중치 로드
        weights_path = 'fault_diagnosis_model.h5'
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"✅ 가중치 로드 완료: {weights_path}")
        else:
            print(f"❌ 오류: 가중치 파일을 찾을 수 없습니다: {weights_path}")
            exit(1)

    # 모델 구조 요약
    print(f"\n📊 모델 구조:")
    model.summary()

    # ========== 4. 테스트 데이터 로드 (1201번부터) ==========
    print("\n📂 Step 4: 테스트 데이터 로드")
    print("-" * 60)

    # CSV 파일 경로
    csv_path = r'D:\AI\PythonProject\.data\car_audio_metadata.csv'

    # CSV 파일 읽기
    car_audio = pd.read_csv(csv_path)
    print(f"✅ CSV 로드 완료!")
    print(f"   - 전체 행 수: {len(car_audio)}")

    # 테스트 데이터 범위 설정 (1201번부터 끝까지)
    test_start_idx = 1200  # 1201번 (인덱스는 0부터 시작하므로 1200)
    test_end_idx = len(car_audio)  # 마지막까지

    # 테스트 데이터 개수 확인
    test_count = test_end_idx - test_start_idx

    # 테스트 데이터가 없으면 종료
    if test_count <= 0:
        print(f"❌ 오류: 테스트 데이터가 없습니다! (전체 데이터: {len(car_audio)}개)")
        print(f"CSV 파일에 1201번 이후 데이터가 있는지 확인하세요.")
        exit(1)

    print(f"\n📊 테스트 데이터 범위:")
    print(f"   - 시작 인덱스: {test_start_idx} (1201번 행)")
    print(f"   - 끝 인덱스: {test_end_idx}")
    print(f"   - 테스트 데이터 개수: {test_count}개")

    # 파일명 추출 (1201번부터)
    mp3_names_test = car_audio.iloc[test_start_idx:test_end_idx, 0]

    # 레이블 추출 (29번 컬럼)
    label_col_idx = label_mapping['label_column_index']  # JSON에서 컬럼 인덱스 가져오기
    labels_test = car_audio.iloc[test_start_idx:test_end_idx, label_col_idx]

    print(f"\n✅ 테스트 데이터 추출 완료!")
    print(f"   - 파일 수: {len(mp3_names_test)}")
    print(f"   - 레이블 컬럼: {car_audio.columns[label_col_idx]}")

    # ========== 5. 레이블 전처리 ==========
    print("\n🔄 Step 5: 레이블 전처리")
    print("-" * 60)

    # NaN 값 제거
    valid_indices = labels_test.notna()  # NaN이 아닌 인덱스
    mp3_names_test = mp3_names_test[valid_indices]
    labels_test = labels_test[valid_indices]

    print(f"NaN 제거 후 데이터 수: {len(labels_test)}")

    # 학습 시 사용한 클래스만 선택 (학습하지 않은 클래스 제외)
    trained_classes = set(label_mapping['encoding'].keys())  # 학습된 클래스 집합

    # 학습된 클래스에 속하는 샘플만 선택
    mask_trained = labels_test.astype(str).isin(trained_classes)  # 문자열로 변환하여 비교

    # 필터링
    mp3_names_test = mp3_names_test[mask_trained]
    labels_test = labels_test[mask_trained]

    print(f"학습된 클래스만 선택 후 데이터 수: {len(labels_test)}")

    # 테스트 데이터가 없으면 종료
    if len(labels_test) == 0:
        print(f"❌ 오류: 학습된 클래스에 해당하는 테스트 데이터가 없습니다!")
        print(f"학습된 클래스: {list(trained_classes)[:10]}...")
        exit(1)

    # 레이블 인코딩 (학습 시와 동일한 매핑 사용)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(list(label_mapping['encoding'].keys()))  # 학습 시 클래스 순서

    # 레이블을 숫자로 변환
    encoded_labels_test = label_encoder.transform(labels_test.astype(str))

    # 클래스 개수
    num_classes = len(label_encoder.classes_)

    print(f"\n✅ 레이블 인코딩 완료!")
    print(f"   - 클래스 개수: {num_classes}")
    print(f"   - 레이블 범위: {np.min(encoded_labels_test)} ~ {np.max(encoded_labels_test)}")

    # 레이블 분포 출력
    print(f"\n테스트 데이터 레이블 분포:")
    unique, counts = np.unique(encoded_labels_test, return_counts=True)
    for label_idx, count in zip(unique, counts):
        class_name = label_encoder.classes_[label_idx]
        print(f"   {class_name}: {count}개")

    # ========== 6. 파일 경로 생성 ==========
    print("\n📁 Step 6: 파일 경로 생성")
    print("-" * 60)

    # 베이스 경로
    base_path = r'D:\AI\PythonProject\.data'

    # 전체 경로 생성 (파일명 앞의 슬래시 제거)
    mp3_paths_test = [
        os.path.join(base_path, str(name).lstrip(r'/\\'))
        for name in mp3_names_test
    ]

    print(f"✅ 파일 경로 생성 완료")
    print(f"   - 베이스 경로: {base_path}")
    print(f"   - 총 파일 수: {len(mp3_paths_test)}")

    # 파일 존재 확인 (처음 3개만)
    print(f"\n샘플 경로 확인 (처음 3개):")
    for i in range(min(3, len(mp3_paths_test))):
        exists = "✅" if os.path.exists(mp3_paths_test[i]) else "❌"
        print(f"   {exists} {mp3_paths_test[i]}")

    # 전체 파일 존재 확인
    existing_count = sum(1 for p in mp3_paths_test if os.path.exists(p))
    print(f"\n파일 존재 확인:")
    print(f"   - 존재: {existing_count}개")
    print(f"   - 없음: {len(mp3_paths_test) - existing_count}개")

    # 파일이 하나도 없으면 종료
    if existing_count == 0:
        print("\n❌ 오류: 존재하는 테스트 파일이 없습니다!")
        print("파일 경로를 확인하세요.")
        exit(1)

    # ========== 7. 오디오 전처리 ==========
    print("\n🎵 Step 7: 테스트 오디오 전처리")
    print("-" * 60)

    # GPU 메모리 확인
    check_gpu_memory()

    # 트레이너 인스턴스 생성 (prepare_data_from_paths 메소드 사용을 위해)
    trainer = FaultDiagnosisTrainer(
        model=model,  # 로드한 모델
        num_classes=num_classes,  # 클래스 개수
        learning_rate=0.001  # 테스트에서는 사용 안 함 (형식상 필요)
    )

    # 오디오 파일들을 스펙트로그램으로 변환
    X_full_test, X_perc_test, y_test = trainer.prepare_data_from_paths(
        audio_paths=mp3_paths_test,  # 테스트 파일 경로
        labels=encoded_labels_test.tolist(),  # 인코딩된 레이블
        target_shape=tuple(model_info['target_shape']),  # 학습 시와 동일한 크기
        use_augmentation=False  # 테스트 시에는 증강 안 함!
    )

    # 데이터 타입 최적화 (float32)
    X_full_test = X_full_test.astype(np.float32)
    X_perc_test = X_perc_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    print(f"\n✅ 전처리 완료!")
    print(f"   - X_full_test: {X_full_test.shape}")
    print(f"   - X_perc_test: {X_perc_test.shape}")
    print(f"   - y_test: {y_test.shape}")

    # GPU 메모리 확인
    check_gpu_memory()

    # ========== 8. 모델 예측 ==========
    print("\n🔮 Step 8: 모델 예측")
    print("-" * 60)

    # 테스트 데이터로 예측 수행
    print("예측 중...")
    y_test_pred_probs = model.predict([X_full_test, X_perc_test])  # 예측 확률

    # 가장 높은 확률의 클래스를 예측값으로
    y_test_pred_classes = np.argmax(y_test_pred_probs, axis=1)  # 예측 클래스

    # 실제 레이블도 클래스 번호로 변환
    y_test_true_classes = np.argmax(y_test, axis=1)  # 실제 클래스

    print(f"✅ 예측 완료!")
    print(f"   - 테스트 샘플 수: {len(y_test_pred_classes)}")

    # ========== 9. 정확도 계산 ==========
    print("\n📊 Step 9: 성능 평가")
    print("-" * 60)

    # 전체 정확도 계산
    test_accuracy = np.mean(y_test_pred_classes == y_test_true_classes)

    print(f"✅ 테스트 정확도: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # 손실 계산 (Categorical Crossentropy)
    loss_fn = keras.losses.CategoricalCrossentropy()  # 손실 함수
    test_loss = loss_fn(y_test, y_test_pred_probs).numpy()  # 손실값

    print(f"✅ 테스트 손실: {test_loss:.4f}")

    # ========== 10. Confusion Matrix 생성 ==========
    print("\n📊 Step 10: Confusion Matrix 생성")
    print("-" * 60)

    # Confusion Matrix 계산
    cm = confusion_matrix(y_test_true_classes, y_test_pred_classes)

    print(f"✅ Confusion Matrix 크기: {cm.shape}")

    # ========== 11. Confusion Matrix 시각화 ==========
    print("\n🎨 Confusion Matrix 시각화 중...")

    # 그래프 크기 설정
    fig_size = max(10, num_classes * 0.8)  # 클래스 개수에 따라 조정
    plt.figure(figsize=(fig_size, fig_size))  # 정사각형

    # 히트맵 그리기
    sns.heatmap(
        cm,  # 혼동 행렬 데이터
        annot=True,  # 숫자 표시
        fmt='d',  # 정수 형식
        cmap='Blues',  # 파란색 계열
        xticklabels=label_encoder.classes_,  # X축 레이블
        yticklabels=label_encoder.classes_,  # Y축 레이블
        cbar_kws={'label': 'Count'},  # 컬러바 레이블
        linewidths=0.5,  # 셀 구분선
        linecolor='gray',  # 구분선 색상
        square=True,  # 정사각형 셀
        annot_kws={"size": 10}  # 숫자 크기
    )

    # 제목 및 축 레이블
    plt.title('Confusion Matrix - Test Set', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('True Label (Actual)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')

    # 레이블 회전
    plt.xticks(rotation=45, ha='right')  # X축 레이블 45도 회전
    plt.yticks(rotation=0)  # Y축 레이블 수평

    # 레이아웃 조정
    plt.tight_layout()

    # 파일로 저장
    plt.savefig('confusion_matrix_test.png', dpi=300, bbox_inches='tight')
    print("✅ Confusion Matrix 저장: confusion_matrix_test.png")

    # 화면에 표시
    plt.show()

    # ========== 12. Classification Report ==========
    print("\n📋 Step 11: Classification Report")
    print("-" * 60)

    # 클래스별 상세 성능 계산
    report = classification_report(
        y_test_true_classes,  # 실제 레이블
        y_test_pred_classes,  # 예측 레이블
        target_names=label_encoder.classes_.astype(str),  # 클래스 이름
        digits=4  # 소수점 4자리
    )

    # 콘솔에 출력
    print("\n" + "=" * 80)
    print("📊 Classification Report (테스트 데이터)")
    print("=" * 80)
    print(report)
    print("=" * 80)

    # 파일로 저장
    with open('classification_report_test.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Classification Report - Test Set\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"테스트 정확도: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)\n")
        f.write(f"테스트 손실: {test_loss:.4f}\n\n")
        f.write(report)
        f.write("\n" + "=" * 80 + "\n")

    print("✅ Classification Report 저장: classification_report_test.txt")

    # ========== 13. 클래스별 정확도 ==========
    print("\n📊 Step 12: 클래스별 정확도")
    print("-" * 60)

    # 각 클래스별 정확도 계산
    class_accuracies = []  # 정확도 저장 리스트

    for i, class_name in enumerate(label_encoder.classes_):
        # 해당 클래스의 샘플 인덱스
        class_indices = (y_test_true_classes == i)

        # 샘플이 없으면 건너뛰기
        if np.sum(class_indices) == 0:
            continue

        # 정확도 계산
        class_acc = np.mean(
            y_test_pred_classes[class_indices] == y_test_true_classes[class_indices]
        )

        # 샘플 개수
        class_count = np.sum(class_indices)

        # 맞춘 개수
        correct_count = np.sum(
            y_test_pred_classes[class_indices] == y_test_true_classes[class_indices]
        )

        # 저장
        class_accuracies.append(class_acc)

        # 출력
        print(f"   클래스 '{class_name}':")
        print(f"      정확도: {class_acc:.4f} ({class_acc * 100:.2f}%)")
        print(f"      맞춘 개수: {correct_count}/{class_count}")
        print()

    # 평균 클래스별 정확도
    mean_class_acc = np.mean(class_accuracies)
    print(f"📊 평균 클래스별 정확도: {mean_class_acc:.4f} ({mean_class_acc * 100:.2f}%)")

    # ========== 14. 오분류 분석 ==========
    print("\n🔍 Step 13: 오분류 분석")
    print("-" * 60)

    # 오분류된 쌍 추출
    misclassifications = []

    for i in range(num_classes):  # 실제 클래스
        for j in range(num_classes):  # 예측 클래스
            if i != j:  # 대각선 제외
                count = cm[i, j]  # 오분류 개수
                if count > 0:
                    misclassifications.append((
                        label_encoder.classes_[i],  # 실제 클래스
                        label_encoder.classes_[j],  # 예측 클래스
                        count  # 개수
                    ))

    # 개수 많은 순으로 정렬
    misclassifications.sort(key=lambda x: x[2], reverse=True)

    # Top 5 출력
    print("가장 많이 오분류된 쌍 (Top 5):")
    for idx, (true_class, pred_class, count) in enumerate(misclassifications[:5], 1):
        print(f"   {idx}. 실제 '{true_class}' → 예측 '{pred_class}': {count}회")

    # 오분류가 없으면
    if len(misclassifications) == 0:
        print("   🎉 오분류 없음! 완벽한 성능!")

    # ========== 15. 신뢰도 분석 ==========
    print("\n🎯 Step 14: 예측 신뢰도 분석")
    print("-" * 60)

    # 각 예측의 최대 확률 (신뢰도)
    confidences = np.max(y_test_pred_probs, axis=1)

    # 통계 계산
    mean_confidence = np.mean(confidences)  # 평균
    min_confidence = np.min(confidences)  # 최소
    max_confidence = np.max(confidences)  # 최대
    median_confidence = np.median(confidences)  # 중간값

    print(f"   평균 신뢰도: {mean_confidence:.4f} ({mean_confidence * 100:.2f}%)")
    print(f"   최소 신뢰도: {min_confidence:.4f} ({min_confidence * 100:.2f}%)")
    print(f"   최대 신뢰도: {max_confidence:.4f} ({max_confidence * 100:.2f}%)")
    print(f"   중간값 신뢰도: {median_confidence:.4f} ({median_confidence * 100:.2f}%)")

    # 신뢰도 구간별 분포
    high_conf = np.sum(confidences >= 0.9)  # 90% 이상
    medium_conf = np.sum((confidences >= 0.7) & (confidences < 0.9))  # 70~90%
    low_conf = np.sum(confidences < 0.7)  # 70% 미만

    print(f"\n   신뢰도 분포:")
    print(f"      높음 (≥90%): {high_conf}개 ({high_conf / len(confidences) * 100:.1f}%)")
    print(f"      중간 (70~90%): {medium_conf}개 ({medium_conf / len(confidences) * 100:.1f}%)")
    print(f"      낮음 (<70%): {low_conf}개 ({low_conf / len(confidences) * 100:.1f}%)")

    # ========== 16. 신뢰도 히스토그램 ==========
    print("\n📊 신뢰도 분포 히스토그램 생성 중...")

    plt.figure(figsize=(10, 6))  # 그래프 크기

    # 히스토그램
    plt.hist(
        confidences,  # 데이터
        bins=20,  # 구간 개수
        color='skyblue',  # 색상
        edgecolor='black',  # 테두리
        alpha=0.7  # 투명도
    )

    # 평균선
    plt.axvline(
        mean_confidence,  # 위치
        color='red',  # 색상
        linestyle='--',  # 점선
        linewidth=2,  # 두께
        label=f'평균: {mean_confidence:.2f}'  # 레이블
    )

    # 제목 및 레이블
    plt.title('Test Prediction Confidence Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()  # 범례
    plt.grid(True, alpha=0.3)  # 격자

    # 저장 및 표시
    plt.tight_layout()
    plt.savefig('confidence_distribution_test.png', dpi=300, bbox_inches='tight')
    print("✅ 신뢰도 히스토그램 저장: confidence_distribution_test.png")
    plt.show()

    # ========== 17. 결과 요약 및 비교 ==========
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)

    print(f"\n학습/검증 성능 (train.py):")
    print(f"   - 학습 정확도: {model_info['train_acc']:.4f} ({model_info['train_acc'] * 100:.2f}%)")
    print(f"   - 검증 정확도: {model_info['val_acc']:.4f} ({model_info['val_acc'] * 100:.2f}%)")
    print(f"   - 검증 손실: {model_info['val_loss']:.4f}")

    print(f"\n테스트 성능 (현재):")
    print(f"   - 테스트 정확도: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"   - 테스트 손실: {test_loss:.4f}")
    print(f"   - 평균 신뢰도: {mean_confidence:.4f} ({mean_confidence * 100:.2f}%)")

    # 성능 차이 계산
    acc_diff = test_accuracy - model_info['val_acc']

    print(f"\n성능 차이:")
    if acc_diff >= 0:
        print(f"   ✅ 테스트 정확도가 검증보다 {abs(acc_diff) * 100:.2f}%p 높음")
    else:
        print(f"   ⚠️ 테스트 정확도가 검증보다 {abs(acc_diff) * 100:.2f}%p 낮음")

    # ========== 18. 파일 목록 ==========
    print("\n📁 생성된 파일:")
    print("   ✅ confusion_matrix_test.png           - 테스트 혼동 행렬")
    print("   ✅ classification_report_test.txt      - 테스트 상세 성능")
    print("   ✅ confidence_distribution_test.png    - 테스트 신뢰도 분포")

    # ========== 19. 최종 메시지 ==========
    print("\n" + "=" * 60)
    print("🎉 테스트 완료!")
    print("=" * 60)

    # 성능 평가
    if test_accuracy >= 0.9:
        print("\n⭐⭐⭐⭐⭐ 탁월한 성능! (90% 이상)")
    elif test_accuracy >= 0.8:
        print("\n⭐⭐⭐⭐ 우수한 성능! (80~90%)")
    elif test_accuracy >= 0.7:
        print("\n⭐⭐⭐ 양호한 성능 (70~80%)")
    else:
        print("\n⚠️ 개선 필요 (70% 미만)")

    print()
