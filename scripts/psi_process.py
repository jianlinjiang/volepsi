import pandas as pd

alicedata = pd.read_csv('../data/A_PSI_DATA.csv')
bobdata = pd.read_csv('../data/B_PSI_DATA.csv')
resdata = pd.read_csv('../data/PSI_RES.csv')
resdata_tmp = pd.read_csv('../data/PSI_RES_TMP.csv')
intersected_df = pd.merge(alicedata, bobdata, how="inner")

diff = pd.concat([resdata, resdata_tmp]).drop_duplicates(keep=False)
print(diff)