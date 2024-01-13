import pandas as pd
df = pd.read_table('/home/yuanshuai20/paper/互作对簇.txt', header=None, names=['col1', 'col2', 'col3'], encoding='utf-8')
df1 = df[['col1', 'col2']]
df_new = pd.DataFrame(df1.to_numpy().reshape(-1, 1, order='F'), columns=['col1'])
print(df_new['col1'].nunique())