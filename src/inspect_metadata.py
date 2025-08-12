# 文件: src/inspect_metadata.py

import pandas as pd
import os

# --- 配置你的文件路径 ---
# 【请确认】这两个路径是正确的
EXCEL_FILE_PATH = "/root/autodl-tmp/ATLAS_data/ATLAS_2/20220425_ATLAS_2.0_MetaData.xlsx"
CSV_FILE_PATH = "/root/autodl-tmp/ATLAS_data/ATLAS_2/20211112_ATLAS_2.0_SupplementaryInfo.csv"

def inspect_excel_file(filepath):
    """读取并展示Excel文件的基本信息"""
    print("=" * 50)
    print(f"正在分析 Excel 文件: {os.path.basename(filepath)}")
    print("=" * 50)

    if not os.path.exists(filepath):
        print(f"错误: 文件未找到 at {filepath}")
        return

    try:
        # 使用 pandas 读取 Excel 文件
        df = pd.read_excel(filepath)
        
        print(f"\n[ 文件基本信息 ]")
        print(f" - 总行数: {len(df)}")
        print(f" - 总列数: {len(df.columns)}")
        
        print(f"\n[ 所有列名 ]")
        # 打印出所有的列名，这样你就能知道该在配置中用哪个
        for col in df.columns:
            print(f" - '{col}'")
            
        print(f"\n[ 文件前 5 行内容预览 ]")
        # .head() 方法可以展示DataFrame的前几行
        print(df.head())

        # 【关键分析】我们来重点看看可能包含病人ID和病灶位置的列
        # 你可以根据实际看到的列名，修改下面的列表
        potential_id_cols = ['ATLAS_ID', 'Subject ID', 'sub_id']
        potential_location_cols = ['Subcortical_or_Cortical', 'Lesion Location', 'Location']
        
        print("\n[ 关键列内容分析 ]")
        for col_name in potential_id_cols:
            if col_name in df.columns:
                print(f" - 分析列 '{col_name}':")
                print(f"   - 前5个值: {df[col_name].head().tolist()}")
                print(f"   - 是否有重复值: {'是' if df[col_name].duplicated().any() else '否'}")
                break
        
        for col_name in potential_location_cols:
            if col_name in df.columns:
                print(f" - 分析列 '{col_name}':")
                # .value_counts() 可以统计每个类别有多少个
                print(f"   - 类别及其数量:\n{df[col_name].value_counts().to_string()}")
                break

    except Exception as e:
        print(f"\n读取或分析文件时出错: {e}")
        print("请确保你已经安装了 `pandas` 和 `openpyxl` (pip install pandas openpyxl)")


def inspect_csv_file(filepath):
    """读取并展示CSV文件的基本信息"""
    print("\n" + "=" * 50)
    print(f"正在分析 CSV 文件: {os.path.basename(filepath)}")
    print("=" * 50)

    if not os.path.exists(filepath):
        print(f"错误: 文件未找到 at {filepath}")
        return

    try:
        # 使用 pandas 读取 CSV 文件
        df = pd.read_csv(filepath)
        
        print(f"\n[ 文件基本信息 ]")
        print(f" - 总行数: {len(df)}")
        print(f" - 总列数: {len(df.columns)}")

        print(f"\n[ 所有列名 ]")
        for col in df.columns:
            print(f" - '{col}'")

        print(f"\n[ 文件前 5 行内容预览 ]")
        print(df.head())

    except Exception as e:
        print(f"\n读取或分析文件时出错: {e}")


if __name__ == '__main__':
    inspect_excel_file(EXCEL_FILE_PATH)
    inspect_csv_file(CSV_FILE_PATH)