# AI 분석 모듈 문서 (analysis/)

## 개요

수집된 엔진 오디오 데이터를 스펙트로그램으로 변환하고, 딥러닝 모델을 통해 엔진 상태를 진단하는 AI 파이프라인입니다.

## 모듈 구성

### 1. audio_preprocessing.py

**역할**: 오디오 신호를 스펙트로그램으로 변환

**주요 클래스**: `AudioPreprocessor`

**파라미터**:
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| sample_rate | 22050 | 샘플링 레이트 (Hz) |
| n_fft | 2048 | FFT 윈도우 크기 |
| hop_length | 512 | STFT 홉 길이 |
| fmax | 6000 | 최대 주파수 (Hz) |

**주요 메서드**:
| 메서드 | 설명 |
|--------|------|
| `load_audio(path)` | MP3/WAV 파일 로드 |
| `normalize_audio(audio)` | -1~1 정규화 |
| `compute_stft(audio)` | STFT 계산 |
| `compute_log_spectrogram(stft)` | 로그 스펙트로그램 생성 |
| `separate_harmonic_percussive(stft)` | HPSS 분리 |
| `preprocess(path)` | 전체 파이프라인 (파일 입력) |
| `preprocess_from_array(array)` | 전체 파이프라인 (배열 입력) |

**출력**:
- `full_spectrogram`: 전체 주파수 스펙트로그램 (0~6000Hz)
- `percussive_spectrogram`: 타악음(노킹 등) 스펙트로그램

### 2. AudioAugment.py

**역할**: 오디오 데이터 증강으로 학습 데이터 다양성 증가

**주요 클래스**: `AudioAugmentor`

**증강 기법**:
| 기법 | 메서드 | 설명 |
|------|--------|------|
| 시간 이동 | `time_shift()` | 오디오를 좌우로 무작위 이동 |
| 피치 변경 | `pitch_shift()` | 음높이 변경 (-2~+2 반음) |
| 속도 변경 | `time_stretch()` | 재생 속도 변경 (0.8~1.2배) |
| 노이즈 추가 | `add_noise()` | 백색 노이즈 추가 |
| 볼륨 변경 | `change_volume()` | 볼륨 증가/감소 (-6~+6 dB) |

**사용 예시**:
```python
augmentor = AudioAugmentor(sample_rate=22050)
augmented = augmentor.apply_random_augmentations(audio, probability=0.5)
```

### 3. conv_blocks.py

**역할**: CNN 기본 블록 정의

**주요 클래스**: `ConvBlock`

**구조**:
```
Conv2D → BatchNormalization → ReLU
```

**파라미터**:
| 파라미터 | 설명 |
|----------|------|
| filters | 출력 채널 수 |
| kernel_size | 필터 크기 (기본 3x3) |
| strides | 이동 간격 |
| padding | 패딩 방식 ('same') |

### 4. attention_modules.py

**역할**: CBAM (Convolutional Block Attention Module) 구현

**주요 클래스**: `CBAM`

**동작 원리**:
1. **채널 어텐션**: 어떤 특징(채널)이 중요한지 학습
   - Global Average Pooling + Global Max Pooling
   - MLP로 중요도 계산

2. **공간 어텐션**: 스펙트로그램의 어느 위치가 중요한지 학습
   - 채널 방향 평균/최대값
   - 7x7 Conv로 중요도 맵 생성

**효과**: 모델이 고장 신호에 더 집중하도록 유도

### 5. inference.py

**역할**: 학습된 모델로 새로운 오디오 진단

**주요 클래스**: `FaultDiagnosisInference`

**사용법**:
```python
inference = FaultDiagnosisInference(
    model_path='fault_diagnosis_model.h5',
    class_names=['정상', '베어링 고장', '기타 소음']
)

predicted_class, probabilities = inference.predict('engine_sound.mp3')
```

**출력 예시**:
```
📊 진단 결과:
----------------------------------------
정상      : 85.32% ██████████████████████████████████████████
베어링 고장:  8.45% ████
기타 소음  :  6.23% ███
----------------------------------------
✅ 최종 진단: 정상 (신뢰도: 85.32%)
```

### 6. TrainModelTest.py

**역할**: 학습된 모델 테스트 및 성능 평가

**주요 기능**:
- 테스트 데이터 로드 (1201번 이후 데이터)
- 모델 예측 수행
- Confusion Matrix 생성 및 시각화
- Classification Report 출력
- 클래스별 정확도 분석
- 오분류 분석
- 신뢰도 분포 분석

**출력 파일**:
- `confusion_matrix_test.png`: 혼동 행렬 히트맵
- `classification_report_test.txt`: 상세 성능 리포트
- `confidence_distribution_test.png`: 예측 신뢰도 분포

### 7. check_data.py

**역할**: 데이터셋 검증 유틸리티

**기능**:
- CSV 파일 구조 확인
- 레이블 분포 확인
- 파일 경로 존재 여부 검사

### 8. data prep.py

**역할**: 오디오 데이터 탐색 및 시각화

**기능**:
- 오디오 파형(Waveform) 시각화
- STFT 스펙트로그램 시각화
- HPSS 분리 결과 시각화 (Harmonic/Percussive)

## 딥러닝 모델 아키텍처

### 이중 입력 CNN + CBAM

```
입력 1: Full Spectrogram (128x128x1)
        │
        ▼
    ConvBlock (64)
        │
        ▼
    ConvBlock (128)
        │
        ▼
      CBAM
        │
        ▼
    MaxPooling
        │
        ├──────────────────────┐
        │                      │
입력 2: Percussive Spectrogram
        │
        ▼
    ConvBlock (64)
        │
        ▼
    ConvBlock (128)
        │
        ▼
      CBAM
        │
        ▼
    MaxPooling
        │
        ├──────────────────────┘
        │
        ▼
    Concatenate
        │
        ▼
    Dense (256)
        │
        ▼
    Dropout (0.5)
        │
        ▼
    Dense (num_classes)
        │
        ▼
    Softmax
```

## 처리 파이프라인

```
1. 오디오 로드 (MP3/WAV)
        │
        ▼
2. 정규화 (-1 ~ 1)
        │
        ▼
3. STFT 변환
        │
        ├── Full Spectrogram
        │
        └── HPSS → Percussive Spectrogram
        │
        ▼
4. dB 변환 및 0~1 정규화
        │
        ▼
5. 크기 조정 (128x128)
        │
        ▼
6. 모델 입력 및 예측
```
