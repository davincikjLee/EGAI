"""
Data Manager - 데이터 저장 및 관리

기능:
    - CSV 메타데이터 관리
    - 파일 시스템 관리
    - 데이터 무결성 검사
"""

import os
from pathlib import Path
from typing import Dict, Any, Set, Optional, List
import pandas as pd


class DataManager:
    """
    데이터 저장 및 관리
    """

    # 메타데이터 컬럼 정의
    METADATA_COLUMNS = [
        "audio_file_path", "data_label", "goodsNo", "vehicle_name",
        "first_registration_date", "year", "current_mileage_km",
        "vehicle_type", "seating_capacity", "fuel_type", "displacement_cc",
        "drivetrain", "transmission_type", "exterior_color", "interior_color",
        "vehicle_number", "popular_package_applied", "certified_inspection_passed",
        "inspection_date", "oil_filter_changed", "ac_filter_changed",
        "wiper_blades_changed", "washer_fluid_replenished",
        "warranty_remaining_km", "warranty_remaining_months",
        "my_car_damage_reported", "owner_changed", "liens_encumbrances_exist",
        "overall_score", "mid_freq_score", "low_high_freq", "audible_range_score",
        "regularity", "irregularity", "specific_anomaly", "jessino"
    ]

    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: 데이터 디렉토리 경로
        """
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / "car_audio_metadata.csv"
        self.goods_nos_path = self.data_dir / "goods_nos.csv"
        self.vehicle_assets_dir = self.data_dir / "vehicle_assets"

        # 디렉토리 생성
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vehicle_assets_dir.mkdir(parents=True, exist_ok=True)

    def get_existing_goods_nos(self) -> Set[str]:
        """기존 수집된 goodsNo 집합 반환"""
        if not self.goods_nos_path.exists():
            return set()

        try:
            df = pd.read_csv(self.goods_nos_path)
            return set(df["goodsNo"].tolist())
        except Exception:
            return set()

    def get_vehicle_asset_dir(self, goods_no: str) -> str:
        """차량 에셋 디렉토리 경로 반환 및 생성"""
        asset_dir = self.vehicle_assets_dir / goods_no
        asset_dir.mkdir(parents=True, exist_ok=True)
        return str(asset_dir)

    def save_metadata(self, data: Dict[str, Any]):
        """메타데이터 CSV에 저장"""
        row = {col: data.get(col) for col in self.METADATA_COLUMNS}
        new_df = pd.DataFrame([row])

        if self.metadata_path.exists():
            existing_df = pd.read_csv(self.metadata_path)

            # 중복 체크 및 업데이트
            if data.get("goodsNo") in existing_df["goodsNo"].values:
                mask = existing_df["goodsNo"] == data["goodsNo"]
                for col in self.METADATA_COLUMNS:
                    if col in row:
                        existing_df.loc[mask, col] = row[col]
            else:
                existing_df = pd.concat([existing_df, new_df], ignore_index=True)

            existing_df.to_csv(self.metadata_path, index=False)
        else:
            new_df.to_csv(self.metadata_path, index=False)

    def save_goods_no(self, goods_no: str, data_collected: bool = False, mp3_downloaded: bool = False):
        """goodsNo 상태 저장"""
        new_row = pd.DataFrame([{
            "goodsNo": goods_no,
            "data_collected": data_collected,
            "mp3_downloaded": mp3_downloaded,
        }])

        if self.goods_nos_path.exists():
            df = pd.read_csv(self.goods_nos_path)
            if goods_no not in df["goodsNo"].values:
                df = pd.concat([df, new_row], ignore_index=True)
            else:
                df.loc[df["goodsNo"] == goods_no, "data_collected"] = data_collected
                df.loc[df["goodsNo"] == goods_no, "mp3_downloaded"] = mp3_downloaded
            df.to_csv(self.goods_nos_path, index=False)
        else:
            new_row.to_csv(self.goods_nos_path, index=False)

    def load_metadata(self, fuel_type: Optional[str] = None) -> pd.DataFrame:
        """
        메타데이터 로드

        Args:
            fuel_type: 연료 타입 필터 (가솔린/디젤)

        Returns:
            메타데이터 DataFrame
        """
        if not self.metadata_path.exists():
            return pd.DataFrame(columns=self.METADATA_COLUMNS)

        df = pd.read_csv(self.metadata_path)

        # 유효 데이터 필터링 (overall_score > 0)
        df = df[df["overall_score"] > 0]

        # 연료 타입 필터
        if fuel_type:
            df = df[df["fuel_type"] == fuel_type]

        return df

    def get_valid_audio_indices(self) -> List[int]:
        """유효한 오디오 파일이 있는 인덱스 반환"""
        if not self.metadata_path.exists():
            return []

        df = pd.read_csv(self.metadata_path)
        valid_indices = []

        for idx, row in df.iterrows():
            if row["overall_score"] > 0:
                audio_path = self.data_dir / row["audio_file_path"]
                if audio_path.exists():
                    valid_indices.append(idx)

        return valid_indices

    def get_data_stats(self) -> Dict[str, Any]:
        """데이터 통계 반환"""
        stats = {
            "total_metadata": 0,
            "valid_samples": 0,
            "with_audio": 0,
            "fuel_types": {},
        }

        if not self.metadata_path.exists():
            return stats

        df = pd.read_csv(self.metadata_path)
        stats["total_metadata"] = len(df)

        valid_df = df[df["overall_score"] > 0]
        stats["valid_samples"] = len(valid_df)

        # 오디오 파일 존재 체크
        with_audio = 0
        for _, row in valid_df.iterrows():
            audio_path = self.data_dir / str(row["audio_file_path"])
            if audio_path.exists():
                with_audio += 1
        stats["with_audio"] = with_audio

        # 연료 타입 분포
        if "fuel_type" in valid_df.columns:
            stats["fuel_types"] = valid_df["fuel_type"].value_counts().to_dict()

        return stats

    def merge_new_data(
        self,
        new_data_dir: str,
        dry_run: bool = True,
    ) -> Dict[str, int]:
        """
        새 데이터 병합

        Args:
            new_data_dir: 새 데이터 디렉토리
            dry_run: True면 시뮬레이션만

        Returns:
            병합 결과 통계
        """
        new_dir = Path(new_data_dir)
        stats = {"total": 0, "new": 0, "duplicate": 0, "merged": 0}

        # 새 메타데이터 찾기
        new_metadata_path = new_dir / "car_audio_metadata.csv"
        if not new_metadata_path.exists():
            print(f"[DataManager] 새 메타데이터 없음: {new_metadata_path}")
            return stats

        new_df = pd.read_csv(new_metadata_path)
        stats["total"] = len(new_df)

        existing_goods_nos = self.get_existing_goods_nos()

        for _, row in new_df.iterrows():
            goods_no = row.get("goodsNo")
            if goods_no in existing_goods_nos:
                stats["duplicate"] += 1
            else:
                stats["new"] += 1
                if not dry_run:
                    # 메타데이터 저장
                    self.save_metadata(row.to_dict())

                    # 오디오 파일 복사
                    audio_path = row.get("audio_file_path")
                    if audio_path:
                        src = new_dir / audio_path
                        dst = self.data_dir / audio_path
                        if src.exists():
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            import shutil
                            shutil.copy2(src, dst)
                            stats["merged"] += 1

        return stats
