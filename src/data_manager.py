# src/data_manager.py

import os
import pandas as pd
from typing import List, Dict, Union, Any


class DataManager:
    """
    크롤링된 데이터를 관리하고 저장하는 클래스입니다.
    CSV 파일 저장, goodsNo 목록 관리, 디버깅 HTML 저장 등을 담당합니다.
    """

    def __init__(self):
        # ... (기존 __init__ 코드는 그대로 유지) ...
        # 현재 스크립트 파일의 디렉토리
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # 프로젝트 루트 디렉토리 (src의 부모 디렉토리)
        self.project_root_dir = os.path.normpath(os.path.join(current_script_dir, '..'))

        self.data_dir = os.path.join(self.project_root_dir, 'data')
        self.debug_html_dir = os.path.join(self.data_dir, 'debug_html')
        self.goods_nos_csv_path = os.path.join(self.data_dir, 'goods_nos.csv')
        self.metadata_csv_path = os.path.join(self.data_dir, 'car_audio_metadata.csv')
        self.vehicle_assets_dir = os.path.join(self.data_dir, 'vehicle_assets')  # MP3 저장 경로

        # 필요한 디렉토리 생성
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.debug_html_dir, exist_ok=True)
        os.makedirs(self.vehicle_assets_dir, exist_ok=True)

        # 초기 설계 메타데이터 컬럼 순서 정의
        self.metadata_columns_order = [
            "audio_file_path", "data_label", "goodsNo", "vehicle_name",
            "first_registration_date", "year", "current_mileage_km",
            "vehicle_type", "seating_capacity", "fuel_type", "displacement_cc",
            "drivetrain", "transmission_type", "exterior_color", "interior_color",
            "vehicle_number", "popular_package_applied", "certified_inspection_passed",
            "inspection_date", "oil_filter_changed", "ac_filter_changed",
            "wiper_blades_changed", "washer_fluid_replenished",
            "warranty_remaining_km", "warranty_remaining_months",
            "my_car_damage_reported", "owner_changed", "liens_encumbrances_exist",
            "overall_score", "mid_freq_score", "low_high_freq", "audable_range_score",
            "regularity", "irregularity", "specific_anomaly", "jessino"
        ]

    def get_base_data_path(self) -> str:
        """데이터 저장 기본 경로를 반환합니다."""
        return self.data_dir

    def load_goods_nos_with_status(self) -> pd.DataFrame:
        """
        goods_nos.csv 파일에서 goodsNo와 처리 상태를 로드합니다.
        파일이 없거나 비어있으면 빈 DataFrame을 반환하고, 필요한 컬럼을 추가합니다.
        """
        if os.path.exists(self.goods_nos_csv_path):
            try:
                # 파일이 존재하면 읽기 시도
                df = pd.read_csv(self.goods_nos_csv_path)

                # 필요한 컬럼이 없으면 추가하고 기본값 설정
                if 'data_collected' not in df.columns:
                    df['data_collected'] = False
                if 'mp3_downloaded' not in df.columns:
                    df['mp3_downloaded'] = False
                return df

            except pd.errors.EmptyDataError:
                # 파일은 있지만 내용이 비어있는 경우 (에러 발생 원인)
                print(f"  [경고] '{self.goods_nos_csv_path}' 파일이 비어 있어 초기화합니다.")
                return pd.DataFrame(columns=['goodsNo', 'data_collected', 'mp3_downloaded'])
        else:
            # 파일이 아예 없는 경우
            return pd.DataFrame(columns=['goodsNo', 'data_collected', 'mp3_downloaded'])

    def save_goods_nos_with_status(self, df: pd.DataFrame):
        """
        goodsNo DataFrame을 goods_nos.csv 파일에 저장합니다.
        """
        df.to_csv(self.goods_nos_csv_path, index=False)

    # ... (나머지 메서드들은 기존 코드 그대로 유지) ...

    def add_new_goods_nos_to_df(self, existing_df: pd.DataFrame,
                                new_goods_nos_list: List[Dict[str, Any]]) -> pd.DataFrame:
        if not new_goods_nos_list:
            return existing_df

        new_df = pd.DataFrame(new_goods_nos_list)
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['goodsNo'], keep='first')

        if 'data_collected' not in combined_df.columns:
            combined_df['data_collected'] = False
        if 'mp3_downloaded' not in combined_df.columns:
            combined_df['mp3_downloaded'] = False

        return combined_df

    def update_goods_no_status(self, df: pd.DataFrame, goods_no: str, column: str, status: bool) -> pd.DataFrame:
        if goods_no in df['goodsNo'].values:
            df.loc[df['goodsNo'] == goods_no, column] = status
        else:
            new_row = pd.DataFrame([{'goodsNo': goods_no, 'data_collected': False, 'mp3_downloaded': False}])
            new_row.loc[0, column] = status
            df = pd.concat([df, new_row], ignore_index=True)
            print(f"  [정보] goodsNo {goods_no}가 goods_nos.csv에 새로 추가되었습니다.")
        return df

    def save_metadata_to_csv(self, data: Dict[str, Any]):
        row_data = {col: data.get(col) for col in self.metadata_columns_order}
        new_df = pd.DataFrame([row_data])

        if os.path.exists(self.metadata_csv_path):
            existing_df = pd.read_csv(self.metadata_csv_path)
            if data['goodsNo'] in existing_df['goodsNo'].values:
                existing_df.loc[existing_df['goodsNo'] == data['goodsNo']] = new_df.values
                print(f"    [정보] goodsNo {data['goodsNo']}의 메타데이터가 업데이트되었습니다.")
            else:
                existing_df = pd.concat([existing_df, new_df], ignore_index=True)
                print(f"    [정보] goodsNo {data['goodsNo']}의 메타데이터가 새로 추가되었습니다.")
            existing_df.to_csv(self.metadata_csv_path, index=False)
        else:
            new_df.to_csv(self.metadata_csv_path, index=False)
            print(f"    [정보] car_audio_metadata.csv 파일이 새로 생성되었습니다.")

    def save_debug_html(self, goods_no: str, html_content: str, filename_suffix: str = ""):
        filename = f"debug_{filename_suffix}_{goods_no}.html" if filename_suffix else f"debug_{goods_no}.html"
        filepath = os.path.join(self.debug_html_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  [디버그] HTML을 '{filepath}'에 저장했습니다.")

    def create_vehicle_asset_dir(self, goods_no: str) -> str:
        asset_dir = os.path.join(self.vehicle_assets_dir, goods_no)
        os.makedirs(asset_dir, exist_ok=True)
        return asset_dir