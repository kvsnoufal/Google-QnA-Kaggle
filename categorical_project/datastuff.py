
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pickle
import keras 
from keras import layers
from keras import models
from keras import callbacks
from keras import backend as K
import pandas as pd
import numpy as np
import os
import datetime
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import LabelEncoder
from keras.layers import BatchNormalization,Dense,Dropout,SpatialDropout1D
import tensorflow as tf
from sklearn import metrics
from keras import optimizers

id_cols=['id']       
target_col=['target']
num_cols=[]
num_instance_moneycols=[]
categorical_cols=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1',
       'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',
       'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month']
numeric_cols=num_cols+num_instance_moneycols

train=pd.read_csv("../input/cat-in-the-dat-ii/train.csv").fillna("xxxx")
test=pd.read_csv("../input/cat-in-the-dat-ii/test.csv").fillna("xxxx")
data=train.append(test)
def auc(y_true, y_pred):
    def fallback_auc(y_true, y_pred):
        try:
            return metrics.roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)

for col in categorical_cols:
    enc=LabelEncoder()
    enc.fit(data[col].astype(str).values.reshape(-1,1))
    train[col]=enc.transform(train[col].astype(str).values.reshape(-1,1)).flatten()
    test[col]=enc.transform(test[col].astype(str).values.reshape(-1,1)).flatten()


X=train[categorical_cols].values
y=train[target_col[0]].values
X_test=test[categorical_cols].values



def create_model():
    inputs=[]
    outputs=[]
    for c in categorical_cols:
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))
        inp = layers.Input(shape=(1,))
        out = layers.Embedding(num_unique_values + 1, embed_dim, name=c)(inp)
        out = layers.SpatialDropout1D(0.4)(out)
        out = layers.Reshape(target_shape=(embed_dim, ))(out)
        inputs.append(inp)
        outputs.append(out)
    x = layers.Concatenate()(outputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(300, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(300, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    y = Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inputs, outputs=y)
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate= 0.00146, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=[auc])
    return model

earlystopping = callbacks.EarlyStopping(monitor='val_auc', min_delta=0,\
                              patience=5, verbose=0, mode='max')
checkpoint = callbacks.ModelCheckpoint('bestmodel.h5', monitor='val_auc', verbose=0, \
                             save_best_only=False, period=1)

rlr = callbacks.ReduceLROnPlateau( monitor='val_auc',\
                                  factor=0.1, patience=3, verbose=0, \
                                  cooldown=0, min_lr=0)
CALLBACKS=[earlystopping,checkpoint,rlr]
NFOLDS=10
EPOCHS=10
BATCHSIZE=64
skf=StratifiedKFold(n_splits=NFOLDS)

predictions=np.zeros((len(test),))
validations=np.zeros((len(train),))
for train_index, valid_index in skf.split(X, y):
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    
    model=create_model()
    model.fit(list(np.transpose(X_train)),y_train,validation_data=(list(np.transpose(X_valid)),y_valid),\
              epochs=EPOCHS,batch_size=BATCHSIZE,verbose=2,callbacks=CALLBACKS)
    validations[valid_index]=model.predict(list(np.transpose(X_valid))).flatten()
    predictions += model.predict(list(np.transpose(X_test))).flatten()/NFOLDS
    

submission=pd.DataFrame(predictions,columns=target_col)
submission[id_cols[0]]=test[id_cols[0]]
submission.to_csv("submission.csv", index = False)
dfvals=pd.DataFrame(validations,columns=target_col)
dfvals[id_cols[0]]=train[id_cols[0]]
dfvals.to_csv("validation.csv", index = False)
