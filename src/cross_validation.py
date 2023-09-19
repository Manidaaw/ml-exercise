from sklearn import model_selection
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

"""
- -- binary classification
- -- multi class classification
- -- multi label classification
- -- single column regression
- -- multi column regression
- -- holdout
"""

class CrossValidation:
    def __init__(
            self,
            df,
            target_cols,
            shuffle,
            problem_type="binary_classification",
            multilabel_delimiter = ",",
            n_folds = 5,
            random_state = 42,
            ):
        self.df = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.multilabel_delimiter = multilabel_delimiter

        if self.shuffle is True:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df["kfold"] = -1
    def split(self):
        if self.problem_type in ["binary_classification", "multiclass_classification"]:
            if self.num_targets > 1:
                raise Exception("Invalid number of targets for this problem type!")
            target = self.target_cols[0]
            unique_value = self.df[target].nunique()
            if unique_value == 1:
                raise Exception("Only one unique value found!")
            elif unique_value > 1:
                kf = model_selection.StratifiedKFold(
                    n_splits=self.n_folds,
                    shuffle=False,
                    )
                for fold, (train_idx, val_idx) in enumerate(kf.split(X = self.df, y = self.df.target.values)):
                    self.df.loc[val_idx, "kfold"] = fold
        elif self.problem_type in ["single_col_regression", "multi_col_regression"]:
            if self.num_targets > 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of targets for this problem type!")
            if self.num_targets < 2 and self.problem_type == "multi_col_regression":
                raise Exception("Invalid number of targets for this problem type!")
            kf = model_selection.KFold(n_splits=self.n_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.df)):
                self.df.loc[val_idx, "kfold"] = fold

        elif self.problem_type.startswith("holdout_"):
            if self.num_targets > 1:
                raise Exception("Invalid number of targets for this problem type!")
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_samples = len(self.df) * holdout_percentage / 100
            self.df.loc[:len(self.df) - num_samples, "kfold"] = 0
            self.df.loc[len(self.df) - num_samples:, "kfold"] = 1

        elif self.problem_type == "multilabel_regression":
            if self.num_targets > 1:
                raise Exception("Invalid number of targets for this problem type!")
            targets = self.df[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(
                    n_splits=self.n_folds,
                    shuffle=False,
                    )
            for fold, (train_idx, val_idx) in enumerate(kf.split(X = self.df, y =targets)):
                    self.df.loc[val_idx, "kfold"] = fold

        else:
            raise Exception("Problem type not understood!")
        
        return self.df

if __name__ == "__main__":
    df = pd.read_csv("input/train_multilabel.csv")
    cv = CrossValidation(df, shuffle="True", target_cols=["attribute_ids"], 
                         problem_type="multilabel_regression",
                         multilabel_delimiter=" ")
    df_fixed = cv.split()
    print(df_fixed.head())
    print(df_fixed.kfold.value_counts())
