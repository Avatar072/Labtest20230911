import warnings
import os
import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from mytoolfunction import generatefolder,SaveDataToCsvfile,SaveDataframeTonpArray,CheckFileExists,splitdatasetbalancehalf,printFeatureCountAndLabelCountInfo

filepath = "D:\\ToN-IoT-Network\\TON_IoT Datasets\\UNSW-ToN-IoT"
today = datetime.date.today()
today = today.strftime("%Y%m%d")
# 在D:\\Labtest20230911\\data\\dataset_original產生天日期的資料夾
generatefolder(filepath + "\\", "dataset_AfterProcessed")
generatefolder(filepath + "\\dataset_AfterProcessed\\", today)

def remove_nan_and_inf(df):
    df = df.dropna(how='any', axis=0, inplace=False)
    inf_condition = (df == np.inf).any(axis=1)
    df = df[~inf_condition]
    return df

def clearDirtyData(df):
    # 读取CSV文件并返回DataFrame
    # df = pd.read_csv(file_path,encoding='cp1252',low_memory=False)
    # df = pd.read_csv(file_path)
    # 将每个列中的 "-" 替换为 NaN
    # df.replace("-", pd.NA, inplace=True)
    # 找到不包含NaN、Infinity和"inf"值和"-"值的行
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    return df

### label encoding
def label_Encoding(label):
    label_encoder = preprocessing.LabelEncoder()
    dataset[label] = label_encoder.fit_transform(dataset[label])
    dataset[label].unique()


def label_encoding(label, dataset):
    label_encoder = preprocessing.LabelEncoder()
    original_values = dataset[label].unique()
    
    dataset[label] = label_encoder.fit_transform(dataset[label])
    encoded_values = dataset[label].unique()
    
    return original_values, encoded_values   


### label Encoding And Replace the number of greater than 10,000
def ReplaceMorethanTenthousandQuantity(df):
  
    # 超過提取10000行的只取10000，其餘保留 
    # df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\total_encoded.csv")
    df = pd.read_csv(filepath + "\\dataset_AfterProcessed\\Train_Test_Network_AfterProcessed.csv")
    # 获取每个标签的出现次数
    label_counts = df['type'].value_counts()
    # 打印提取后的DataFrame
    print(label_counts)
    # 创建一个空的DataFrame来存储结果
    extracted_df = pd.DataFrame()

    # 获取所有不同的标签
    unique_labels = df['type'].unique()

    # 遍历每个标签
    for label in unique_labels:
        # 选择特定标签的行
        label_df = df[df['type'] == label]
    
        # 如果标签的数量超过1万，提取前1万行；否则提取所有行
        if len(label_df) > 10000:
            label_df = label_df.head(10000)
    
        # 将结果添加到提取的DataFrame中
        extracted_df = pd.concat([extracted_df, label_df])

    # 将更新后的DataFrame保存到文件
    # SaveDataToCsvfile(extracted_df, "./data/dataset_AfterProcessed","total_encoded_updated_10000")

    # 打印修改后的结果
    print(extracted_df['type'].value_counts())
    return extracted_df

# Loading datasets
dataset = pd.read_csv(filepath + "\\Train_Test_Network.csv")
generatefolder(filepath, "\\dataset_AfterProcessed")
dataset = clearDirtyData(dataset)

# 列出每个列的最常见值
# most_common_values = dataset.mode().iloc[0]
# 列出每个列除了 "-" 以外的最常见值
most_common_values = dataset.apply(lambda col: col[col != "-"].mode().iloc[0] if any(col != "-") else "-", axis=0)

# 将每列中的 "-" 替换为最常见值
for column in dataset.columns:
    most_common_value = most_common_values[column]
    dataset[column].replace("-", most_common_value, inplace=True)

# 打印替换后的结果
print(dataset)
# 打印结果
print(most_common_values)

if(CheckFileExists(filepath + "\\dataset_AfterProcessed\\Train_Test_Network_AfterProcessed.csv")!=True):
    #存将每列中的 "-" 替换为最常见值後的csv
    dataset.to_csv(filepath + "\\dataset_AfterProcessed\\Train_Test_Network_AfterProcessed.csv", index=False)
else:
    dataset= pd.read_csv(filepath + "\\dataset_AfterProcessed\\Train_Test_Network_AfterProcessed.csv")


dataset = ReplaceMorethanTenthousandQuantity(dataset)
label_Encoding('src_ip')
label_Encoding('src_port')
label_Encoding('dst_ip')
label_Encoding('dst_port')
label_Encoding('proto')
label_Encoding('ts')
label_Encoding('service')
label_Encoding('conn_state')
# 需要做label_Encoding的欄位
label_Encoding('dns_query')
label_Encoding('dns_AA')
label_Encoding('dns_RD')
label_Encoding('dns_RA')
label_Encoding('dns_rejected')
label_Encoding('ssl_version')
label_Encoding('ssl_cipher')
label_Encoding('ssl_resumed')
label_Encoding('ssl_established')
label_Encoding('ssl_subject')
label_Encoding('ssl_issuer')
# label_Encoding('http_trans_depth')
label_Encoding('http_method')
label_Encoding('http_uri')
# label_Encoding('http_version')
label_Encoding('http_user_agent')
label_Encoding('http_orig_mime_types')
label_Encoding('http_resp_mime_types')
label_Encoding('weird_name')
# label_Encoding('weird_addl')
label_Encoding('weird_notice')
# 需要做label_Encoding的欄位
# label_Encoding('type')
# Assuming "type" is the column you want to label encode
original_type_values, encoded_type_values = label_encoding("type", dataset)

print("Original Type Values:", original_type_values)
print("Encoded Type Values:", encoded_type_values)

# dataset.to_csv(filepath + "\\dataset_AfterProcessed\\Train_Test_Network_AfterProcessed_updated_10000.csv", index=False)
# afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\Train_Test_Network_AfterProcessed_updated_10000.csv")


if(CheckFileExists(filepath + "\\dataset_AfterProcessed\\Train_Test_Network_AfterProcessed_updated_10000.csv")!=True):
    dataset.to_csv(filepath + "\\dataset_AfterProcessed\\Train_Test_Network_AfterProcessed_updated_10000.csv", index=False)
    afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\Train_Test_Network_AfterProcessed_updated_10000.csv")

else:
    afterprocess_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\Train_Test_Network_AfterProcessed_updated_10000.csv")


### extracting features
#除了type外的特徵
crop_dataset=afterprocess_dataset.iloc[:,:-1]
# 列出要排除的列名
columns_to_exclude = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'ts']
# 使用条件选择不等于这些列名的列
doScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
undoScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col  in columns_to_exclude]]
# print(doScalerdataset.info)
# print(afterprocess_dataset.info)
# print(undoScalerdataset.info)
X=doScalerdataset
X=X.values
# scaler = preprocessing.StandardScaler() #資料標準化
scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
scaler.fit(X)
X=scaler.transform(X)


## 重新合並MinMax後的特徵
number_of_components=38 # 原45個的特徵，扣掉'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 'Protocol', 'Timestamp', 'type' | 45-7 =38
columns_array=[]
for i in range (number_of_components):
    columns_array.append("principal_Component"+str(i+1))
    
principalComponents = X
principalDf = pd.DataFrame(data = principalComponents
              , columns = columns_array)

finalDf = pd.concat([undoScalerdataset,principalDf, afterprocess_dataset[['type']]], axis = 1)
# print(finalDf)
afterprocess_dataset=finalDf

train_dataframes, test_dataframes = train_test_split(afterprocess_dataset, test_size=0.2, random_state=42)#test_size=0.2表示将数据集分成测试集的比例为20%



# train_dataframes = clearDirtyData(train_dataframes)
# test_dataframes = clearDirtyData(test_dataframes)
label_counts = test_dataframes['type'].value_counts()
print("test_dataframes\n", label_counts)
label_counts = train_dataframes['type'].value_counts()
print("train_dataframes\n", label_counts)

# split train_dataframes各一半
train_half1,train_half2 = splitdatasetbalancehalf(train_dataframes,'type')

# 找到train_df_half1和train_df_half2中重复的行
duplicates = train_half2[train_half2.duplicated(keep=False)]

# 删除train_df_half2中与train_df_half1重复的行
train_df_half2 = train_half2[~train_half2.duplicated(keep=False)]

# train_df_half1和train_df_half2 detail information
printFeatureCountAndLabelCountInfo(train_half1, train_df_half2,'type')



SaveDataToCsvfile(train_dataframes, f"./TON_IoT Datasets/UNSW-ToN-IoT/dataset_AfterProcessed/{today}", f"train_dataframes_{today}")
SaveDataToCsvfile(test_dataframes, f"./TON_IoT Datasets/UNSW-ToN-IoT/dataset_AfterProcessed/{today}", f"test_dataframes_{today}")
SaveDataToCsvfile(train_half1, f"./TON_IoT Datasets/UNSW-ToN-IoT/dataset_AfterProcessed/{today}", f"train_half1_{today}")
SaveDataToCsvfile(train_half2,  f"./TON_IoT Datasets/UNSW-ToN-IoT/dataset_AfterProcessed/{today}", f"train_half2_{today}") 
# train_dataframes.to_csv(filepath + "\\dataset_AfterProcessed\\Train_Test_Network_AfterProcessed_updated_train_dataframes.csv", index=False)
# test_dataframes.to_csv(filepath + "\\dataset_AfterProcessed\\Train_Test_Network_AfterProcessed_updated_test_dataframes.csv", index=False)
SaveDataframeTonpArray(test_dataframes, f"./TON_IoT Datasets/UNSW-ToN-IoT/dataset_AfterProcessed/{today}", "test_ToN-IoT",today)
SaveDataframeTonpArray(train_dataframes, f"./TON_IoT Datasets/UNSW-ToN-IoT/dataset_AfterProcessed/{today}", "train_ToN-IoT",today)
SaveDataframeTonpArray(train_half1, f"./TON_IoT Datasets/UNSW-ToN-IoT/dataset_AfterProcessed/{today}", "train_half1_ToN-IoT", today)
SaveDataframeTonpArray(train_half2, f"./TON_IoT Datasets/UNSW-ToN-IoT/dataset_AfterProcessed/{today}", "train_half2_ToN-IoT", today)