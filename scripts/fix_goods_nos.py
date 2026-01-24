"""goods_nos.csv에 exclude_reason 컬럼 추가 및 문제 차량 표시"""
import pandas as pd
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
data_dir = project_root / "data"

# 1. goods_nos.csv 로드
goods_df = pd.read_csv(data_dir / "goods_nos.csv")

# 2. exclude_reason 컬럼 추가 (없으면)
if 'exclude_reason' not in goods_df.columns:
    goods_df['exclude_reason'] = ''

# 3. 문제 차량들 표시
metadata_df = pd.read_csv(data_dir / "car_audio_metadata.csv")

# 점수가 0이거나 NaN인 차량 (엔진 오디오 섹션 없음)
invalid_mask = (
    (metadata_df['overall_score'] == 0) |
    (metadata_df['overall_score'].isna()) |
    (metadata_df['mid_freq_score'].isna())
)
invalid_scores = metadata_df[invalid_mask]['goodsNo'].tolist()

print(f'점수 이상 차량: {len(invalid_scores)}개')

# exclude_reason 설정
for gn in invalid_scores:
    mask = goods_df['goodsNo'] == gn
    if mask.any():
        goods_df.loc[mask, 'exclude_reason'] = 'no_audio_section'

# 4. 저장
goods_df.to_csv(data_dir / "goods_nos.csv", index=False)

print('goods_nos.csv 업데이트 완료!')
print()
excluded = goods_df[goods_df['exclude_reason'].astype(str).str.len() > 0]
print(f'전체: {len(goods_df)}')
print(f'제외됨: {len(excluded)}')
print()
print('제외된 차량:')
print(excluded[['goodsNo', 'exclude_reason']])
