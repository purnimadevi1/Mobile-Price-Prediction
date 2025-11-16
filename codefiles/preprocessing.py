import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def load_and_preprocessor_data(filepath):
    dataset= pd.read_csv(filepath)
    X=dataset.iloc[:,:-1]
    y=dataset.iloc[:,-1]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    X_train=pd.DataFrame(X_train,columns=X.columns)
    X_test=pd.DataFrame(X_test,columns=X.columns)
    numerical_features =['battery_power','clock_speed','fc','int_memory','m_dep','mobile_wt','n_cores','pc','px_height','px_width','ram','sc_h','sc_w','talk_time']
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', StandardScaler(), numerical_features)
        ],
        remainder = 'passthrough'
    )
    pipeline=Pipeline(
        steps=[('preprocessor',preprocessor)]
    )
    pipeline.fit(X_train)
    X_train = pipeline.transform(X_train)
    X_test = pipeline.transform(X_test)
    return pipeline,X_train,y_train,X_test,y_test