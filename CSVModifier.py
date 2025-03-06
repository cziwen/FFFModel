from operator import concat
from os import remove

import pandas as pd


def add_column (csv_file, column_name, default_value=None, output_file=None):
    df = pd.read_csv (csv_file)
    df[column_name] = default_value  # 添加新列，默认值可以为空或自定义
    output_file = output_file or csv_file  # 直接覆盖或另存
    df.to_csv (output_file, index=False)
    print (f"Column '{column_name}' added to {output_file}.")


def remove_column (csv_file, column_name, output_file=None):
    df = pd.read_csv (csv_file)
    if column_name in df.columns:
        df.drop (columns=[column_name], inplace=True)
        output_file = output_file or csv_file
        df.to_csv (output_file, index=False)
        print (f"Column '{column_name}' removed from {output_file}.")
    else:
        print (f"Column '{column_name}' not found in {csv_file}.")

def concatenate_csv(file1, file2, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df_concat = pd.concat([df1, df2], ignore_index=True)  # 纵向拼接
    df_concat.to_csv(output_file, index=False)
    print(f"Files '{file1}' and '{file2}' concatenated into {output_file}.")



# concatenate_csv("train_data_4321.csv", "train_data_0.csv", "数据/train_data_full.csv")
# remove_column("数据/train_data_full.csv", "Weekday")
