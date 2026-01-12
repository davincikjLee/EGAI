# check_data.py
import pandas as pd
import os

# CSV 파일 로드
csv_path = r'D:\AI\PythonProject\.data\car_audio_metadata.csv'
car_audio = pd.read_csv(csv_path)

print("="*60)
print("📊 CSV 파일 정보")
print("="*60)

print(f"\n전체 행 수: {len(car_audio)}")
print(f"컬럼: {car_audio.columns.tolist()}")

print(f"\n처음 10개 행:")
print(car_audio.head(10))

print(f"\n레이블 분포 (2번째 컬럼):")
print(car_audio.iloc[:, 1].value_counts())

# 파일 경로 생성 및 확인
mp3_names = car_audio.iloc[:100, 0]  # 처음 100개만 테스트
base_path = r'C:\Users\5901818\PycharmProjects\PythonProject\.data'
mp3_paths = [os.path.join(base_path, str(name).lstrip(r'/\\')) for name in mp3_names]

print(f"\n🔍 파일 존재 확인 (처음 100개):")
existing = 0
missing = 0
missing_files = []

for path in mp3_paths:
    if os.path.exists(path):
        existing += 1
    else:
        missing += 1
        missing_files.append(path)

print(f"   ✅ 존재: {existing}개")
print(f"   ❌ 없음: {missing}개")

if missing > 0:
    print(f"\n없는 파일 (처음 5개):")
    for f in missing_files[:5]:
        print(f"   - {f}")
