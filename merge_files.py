import os

import pandas

MY_FOLDER = 'arancino-dataset'

if __name__ == '__main__':

    const_columns = None
    final_dfs = {}

    if os.path.exists(MY_FOLDER):
        for subfolder in [x[0] for x in os.walk(MY_FOLDER) if x[0] != MY_FOLDER]:
            dataframes = []
            print(subfolder)
            for file in os.listdir(subfolder):
                if file.endswith(".csv"):
                    dataframes.append(pandas.read_csv(os.path.join(subfolder, file)))
            df = pandas.concat(dataframes)
            df.to_csv(subfolder + "_all.csv", index=False)
            final_dfs[subfolder + "_all.csv"] = df
            cc = df.columns[df.nunique() == 1].to_numpy()
            if const_columns is None:
                const_columns = cc
            else:
                const_columns = [value for value in const_columns if value in cc]

        for df_tag in final_dfs:
            df = final_dfs[df_tag]
            df_filtered = df.drop(columns=const_columns)
            y = df_filtered['label'].to_numpy()
            df_filtered = df_filtered.select_dtypes(exclude=['object'])
            df_filtered['label'] = y
            df_filtered.to_csv(df_tag.replace('_all', '_filtered'), index=False)

