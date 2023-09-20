from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
"""
- label encoding
- one hot encoding
- binarization

"""

class CategoricalFeature:
    def __init__(
            self,
            df,
            categorical_features,
            encoding_type,
            handle_na=False
            ):
        """
        df: pandas dataframe
        categorical_features: list of column names, e.g. ["ord_1", "nom_0"......]
        encoding_type: label, binary, ohe
        handle_na: True/False
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")
        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        for c in self.cat_feats:
            label = preprocessing.LabelEncoder()
            label.fit(self.df[c].values)
            self.output_df.loc[:, c] = label.transform(self.df[c].values)
            self.label_encoders[c] = label
        return self.output_df
    
    def _binary_encoding(self):
        for c in self.cat_feats:
            label = preprocessing.LabelBinarizer()
            label.fit(self.df[c].values)
            val = label.transform(self.df[c].values)
            self.output_df.drop(c, axis = 1, inplace=True)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[c] = label
        return self.output_df

    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)
        return ohe.transform(self.df[self.cat_feats].values)
        
    def _fit_transform(self):
        
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._binary_encoding()
        elif self.enc_type == "ohe":
            return self._one_hot()
        else:
            raise Exception("Encoding type not understood")
        
    def _transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-9999999")

        if self.enc_type == "label":
            for c, label in self.label_encoders.items():
                dataframe.loc[:, c] = label.transform(dataframe[c].values)
            return dataframe
        
        elif self.enc_type == "binary":
            for c, label in self.binary_encoders.items():
                val = label.transform(dataframe[c].values)
                dataframe.drop(c, axis = 1, inplace=True)
                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin_{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe
        
        else:
            raise Exception("Encoding type not understood")
        
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn import linear_model
    train_df = pd.read_csv("../input/train_cat.csv")
    test_df = pd.read_csv("../input/test_cat.csv")
    # train_idx = train_df["id"].values
    # test_idx = test_df["id"].values
    train_len = len(train_df)
    test_df["target"] = -1
    full_df = pd.concat([train_df, test_df])
    cols = [c for c in train_df.columns if c not in ["id", "target"]]
    cat_feats = CategoricalFeature(full_df, 
                                   categorical_features=cols,
                                   encoding_type="ohe",
                                   handle_na=True
                                   )
    ful_df_transformed = cat_feats._fit_transform()
    # train_df_transformed = ful_df_transformed[ful_df_transformed["id"].isin(train_idx)].reset_index(drop=True)
    # test_df_transformed = ful_df_transformed[ful_df_transformed["id"].isin(test_idx)].reset_index(drop=True)
    train_df_transformed = ful_df_transformed[:train_len, :]
    test_df_transformed = ful_df_transformed[train_len:, :]
    
    clf = linear_model.LogisticRegression()
    clf.fit(train_df_transformed, train_df.target.values)
    predictions = clf.predict_proba(test_df_transformed)[:, 1]

    submission = pd.DataFrame(np.column_stack((test_df.id, predictions)), columns=["id", "target"])
    submission.id = submission.id.astype(int)
    submission.to_csv("../input/submission_cat.csv", index=False)



