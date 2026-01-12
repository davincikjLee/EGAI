"""
점수 관계 분석
목표: 5개 개별 점수와 overall_score 간의 관계 파악
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "car_audio_metadata.csv"


def load_data():
    """데이터 로드 및 유효 데이터 필터링"""
    df = pd.read_csv(DATA_PATH)

    # 유효 데이터 (overall_score > 0)
    df_valid = df[df['overall_score'] > 0].copy()

    # 점수 컬럼
    score_columns = ['mid_freq_score', 'irregularity', 'low_high_freq', 'regularity', 'overall_score']

    return df_valid, score_columns


def analyze_correlations(df, score_columns):
    """상관관계 분석"""
    print("=" * 60)
    print("1. 상관관계 분석")
    print("=" * 60)

    scores = df[score_columns]
    corr_matrix = scores.corr()

    print("\n상관계수 행렬:")
    print(corr_matrix.round(3).to_string())

    print("\n\noverall_score와의 상관계수:")
    for col in score_columns[:-1]:
        corr = corr_matrix.loc[col, 'overall_score']
        print(f"  {col:20s}: {corr:+.4f}")

    return corr_matrix


def analyze_statistics(df, score_columns):
    """기초 통계 분석"""
    print("\n" + "=" * 60)
    print("2. 기초 통계")
    print("=" * 60)

    stats = df[score_columns].describe()
    print(stats.round(3).to_string())

    # 값 분포
    print("\n\n값 분포 (빈도):")
    for col in score_columns:
        print(f"\n{col}:")
        value_counts = df[col].value_counts().sort_index()
        for val, count in value_counts.items():
            pct = count / len(df) * 100
            bar = "#" * int(pct / 2)
            print(f"  {val}: {count:3d} ({pct:5.1f}%) {bar}")


def test_simple_formulas(df, score_columns):
    """간단한 공식 테스트"""
    print("\n" + "=" * 60)
    print("3. 간단한 공식 테스트")
    print("=" * 60)

    X_cols = score_columns[:-1]
    X = df[X_cols].values
    y = df['overall_score'].values

    formulas = {
        "평균": lambda x: x.mean(axis=1),
        "가중평균 (균등)": lambda x: x.mean(axis=1),
        "최솟값": lambda x: x.min(axis=1),
        "최댓값": lambda x: x.max(axis=1),
        "중앙값": lambda x: np.median(x, axis=1),
        "조화평균": lambda x: len(X_cols) / np.sum(1/x, axis=1),
        "기하평균": lambda x: np.prod(x, axis=1) ** (1/len(X_cols)),
    }

    print(f"\n{'공식':<20} {'MAE':>8} {'R²':>8} {'RMSE':>8}")
    print("-" * 50)

    best_mae = float('inf')
    best_formula = None

    for name, formula in formulas.items():
        try:
            y_pred = formula(X)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            print(f"{name:<20} {mae:>8.4f} {r2:>8.4f} {rmse:>8.4f}")

            if mae < best_mae:
                best_mae = mae
                best_formula = name
        except Exception as e:
            print(f"{name:<20} 오류: {e}")

    print(f"\n최적 공식: {best_formula} (MAE={best_mae:.4f})")

    # 정확히 일치하는 비율 확인
    y_avg = X.mean(axis=1)
    exact_match = np.sum(np.isclose(y, y_avg, atol=0.01))
    print(f"\n평균과 정확히 일치: {exact_match}/{len(y)} ({exact_match/len(y)*100:.1f}%)")


def linear_regression_analysis(df, score_columns):
    """선형 회귀 분석"""
    print("\n" + "=" * 60)
    print("4. 선형 회귀 분석")
    print("=" * 60)

    X_cols = score_columns[:-1]
    X = df[X_cols].values
    y = df['overall_score'].values

    # Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. 단순 선형 회귀
    print("\n4.1 단순 선형 회귀 (y = a*x1 + b*x2 + c*x3 + d*x4 + e)")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print("\n회귀 계수:")
    for col, coef in zip(X_cols, lr.coef_):
        print(f"  {col:20s}: {coef:+.4f}")
    print(f"  {'절편 (intercept)':20s}: {lr.intercept_:+.4f}")

    # 계수 합 확인
    coef_sum = sum(lr.coef_)
    print(f"\n계수 합: {coef_sum:.4f}")

    y_pred = lr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nTest 성능: MAE={mae:.4f}, R²={r2:.4f}")

    # 2. 계수 합 = 1 제약 (평균 형태)
    print("\n4.2 계수 합 = 1 제약 회귀")
    # 수동으로 평균 형태 가정
    # overall = w1*x1 + w2*x2 + w3*x3 + w4*x4, where sum(w) = 1

    # Ridge 회귀로 안정적인 계수 추정
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_train, y_train)

    # 계수 정규화 (합 = 1)
    normalized_coefs = ridge.coef_ / ridge.coef_.sum()

    print("\n정규화된 가중치 (합=1):")
    for col, coef in zip(X_cols, normalized_coefs):
        print(f"  {col:20s}: {coef:.4f} ({coef*100:.1f}%)")

    # 3. 다항 회귀 (2차)
    print("\n4.3 다항 회귀 (2차 항 포함)")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    lr_poly = Ridge(alpha=1.0)  # 과적합 방지
    lr_poly.fit(X_poly_train, y_train)

    y_pred_poly = lr_poly.predict(X_poly_test)
    mae_poly = mean_absolute_error(y_test, y_pred_poly)
    r2_poly = r2_score(y_test, y_pred_poly)
    print(f"Test 성능: MAE={mae_poly:.4f}, R²={r2_poly:.4f}")

    return lr


def mlp_analysis(df, score_columns):
    """MLP로 관계 학습"""
    print("\n" + "=" * 60)
    print("5. MLP 분석 (5개 점수 → overall)")
    print("=" * 60)

    X_cols = score_columns[:-1]
    X = df[X_cols].values.astype(np.float32)
    y = df['overall_score'].values.astype(np.float32)

    # Train/Val/Test 분할
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"\n데이터: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 모델 정의
    model = keras.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='linear')
    ], name='ScoreMLP')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mae']
    )

    print(f"모델 파라미터: {model.count_params()}")

    # 학습
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
        ]
    )

    # 평가
    y_pred = model.predict(X_test, verbose=0).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nMLP Test 성능: MAE={mae:.4f}, R²={r2:.4f}")

    # 예측 vs 실제 비교
    print("\n샘플 비교 (실제 vs 예측):")
    for i in range(min(10, len(y_test))):
        diff = y_pred[i] - y_test[i]
        print(f"  실제: {y_test[i]:.2f}, 예측: {y_pred[i]:.2f}, 차이: {diff:+.2f}")

    return model


def find_exact_formula(df, score_columns):
    """정확한 공식 탐색"""
    print("\n" + "=" * 60)
    print("6. 정확한 공식 탐색")
    print("=" * 60)

    X_cols = score_columns[:-1]
    X = df[X_cols].values
    y = df['overall_score'].values

    # 단순 평균 테스트
    y_avg = X.mean(axis=1)
    diff = y - y_avg

    print("\noverall_score - 평균(4개 점수) 차이 분포:")
    print(f"  최소: {diff.min():.4f}")
    print(f"  최대: {diff.max():.4f}")
    print(f"  평균: {diff.mean():.4f}")
    print(f"  표준편차: {diff.std():.4f}")

    # 차이가 0인 비율
    zero_diff = np.sum(np.abs(diff) < 0.001)
    print(f"\n차이가 0인 샘플: {zero_diff}/{len(y)} ({zero_diff/len(y)*100:.1f}%)")

    # 반올림 테스트
    y_avg_round = np.round(y_avg)
    y_avg_round1 = np.round(y_avg, 1)

    exact_round0 = np.sum(y == y_avg_round)
    exact_round1 = np.sum(np.abs(y - y_avg_round1) < 0.01)

    print(f"\n정수 반올림 일치: {exact_round0}/{len(y)} ({exact_round0/len(y)*100:.1f}%)")
    print(f"소수점 1자리 반올림 일치: {exact_round1}/{len(y)} ({exact_round1/len(y)*100:.1f}%)")

    # 불일치 샘플 분석
    mismatch_idx = np.where(np.abs(diff) > 0.1)[0]
    if len(mismatch_idx) > 0:
        print(f"\n불일치 샘플 ({len(mismatch_idx)}개) 예시:")
        for idx in mismatch_idx[:5]:
            row = df.iloc[idx]
            scores = [row[col] for col in X_cols]
            avg = np.mean(scores)
            print(f"  점수: {scores} → 평균: {avg:.2f}, overall: {row['overall_score']:.2f}")


def main():
    # 데이터 로드
    df, score_columns = load_data()
    print(f"유효 데이터: {len(df)}개")
    print(f"점수 컬럼: {score_columns}")

    # 1. 상관관계 분석
    corr_matrix = analyze_correlations(df, score_columns)

    # 2. 기초 통계
    analyze_statistics(df, score_columns)

    # 3. 간단한 공식 테스트
    test_simple_formulas(df, score_columns)

    # 4. 선형 회귀 분석
    lr_model = linear_regression_analysis(df, score_columns)

    # 5. MLP 분석
    mlp_model = mlp_analysis(df, score_columns)

    # 6. 정확한 공식 탐색
    find_exact_formula(df, score_columns)

    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
