print("pushed")
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
import keras
import pandas as pd
import numpy as np
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,roc_auc_score,f1_score
from  sklearn.model_selection import KFold 
from keras.callbacks import Callback
import gensim
from scipy.stats import spearmanr, rankdata
train=pd.read_csv("../input/google-quest-challenge/train.csv")
test=pd.read_csv("../input/google-quest-challenge/test.csv")
import tensorflow_hub as hub
module_url = "../input/universalsentenceencoderlarge4/"
embed = hub.load(module_url)

# model = gensim.models.KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin', binary=True)
class SpearmanRhoCallback(Callback):
    def __init__(self, training_data, validation_data, patience, model_name):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
        self.patience = patience
        self.value = -1
        self.bad_epochs = 0
        self.model_name = model_name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        rho_val = np.mean([spearmanr(self.y_val[:, ind], y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0])).correlation for ind in range(y_pred_val.shape[1])])
        if rho_val >= self.value:
            self.value = rho_val
            self.model.save_weights(self.model_name)
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            print("Epoch %05d: early stopping Threshold" % epoch)
            self.model.stop_training = True
        print('\rval_spearman-rho: %s' % (str(round(rho_val, 4))), end=100*' '+'\n')
        return rho_val

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

ids=["qa_id"]
targets=['question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written', 'answer_helpful',
       'answer_level_of_information', 'answer_plausible', 'answer_relevance',
       'answer_satisfaction', 'answer_type_instructions',
       'answer_type_procedure', 'answer_type_reason_explanation',
       'answer_well_written']
feats=[ 'question_title', 'question_body', 'question_user_name',
       'question_user_page', 'answer', 'answer_user_name', 'answer_user_page',
       'url', 'category', 'host']
text_feats=['question_title', 'question_body','answer']
NFOLDS=10
kf = KFold(n_splits=NFOLDS)
kf.get_n_splits(train.qa_id)
print(text_feats)
for train_index, test_index in kf.split(train):
    print("TRAIN:", train_index, "TEST:", test_index)
    qt_train=train.loc[train_index,"question_title"].values
    qb_train=train.loc[train_index,'question_body'].values
    aw_train=train.loc[train_index,'answer'].values
    qt_valid=train.loc[test_index,"question_title"].values
    qb_valid=train.loc[test_index,'question_body'].values
    aw_valid=train.loc[test_index,'answer'].values
    
    y_train=train.loc[train_index,targets].values
    y_valid=train.loc[test_index,targets].values
    break
qt_test=test["question_title"].values
qb_test=test["question_body"].values
aw_test=test["answer"].values


# token = text.Tokenizer(num_words=None)
# max_len = 150

# token.fit_on_texts(list(qt_train) + list(qb_train) + list(aw_train)+\
#                   list(qt_valid) + list(qb_valid) + list(aw_valid)+\
#                   list(qt_test) + list(qb_test) + list(aw_test))


# qttrain_seq = token.texts_to_sequences(qt_train)
# qtvalid_seq = token.texts_to_sequences(qt_valid)
# qttest_seq = token.texts_to_sequences(qt_test)

# qbtrain_seq = token.texts_to_sequences(qb_train)
# qbvalid_seq = token.texts_to_sequences(qb_valid)
# qbtest_seq = token.texts_to_sequences(qb_test)

# awtrain_seq = token.texts_to_sequences(aw_train)
# awvalid_seq = token.texts_to_sequences(aw_valid)
# awtest_seq = token.texts_to_sequences(aw_test)

# # # zero pad the sequences
# qttrain_pad = sequence.pad_sequences(qttrain_seq, maxlen=max_len)
# qtvalid_pad = sequence.pad_sequences(qtvalid_seq, maxlen=max_len)
# qttest_pad = sequence.pad_sequences(qttest_seq, maxlen=max_len)

# qbtrain_pad = sequence.pad_sequences(qbtrain_seq, maxlen=max_len)
# qbvalid_pad = sequence.pad_sequences(qbvalid_seq, maxlen=max_len)
# qbtest_pad = sequence.pad_sequences(qbtest_seq, maxlen=max_len)

# awtrain_pad = sequence.pad_sequences(awtrain_seq, maxlen=max_len)
# awvalid_pad = sequence.pad_sequences(awvalid_seq, maxlen=max_len)
# awtest_pad = sequence.pad_sequences(awtest_seq, maxlen=max_len)

# word_index = token.word_index

qtusetest=embed(qt_test)["outputs"].numpy()
qbusetest=embed(qb_test)["outputs"].numpy()
awusetest=embed(aw_test)["outputs"].numpy()
use_test=[qtusetest,qbusetest,awusetest]

from tqdm import tqdm
# create an embedding matrix for the words we have in the dataset
# embedding_matrix = np.zeros((len(word_index) + 1, 300))
# for word, i in tqdm(word_index.items()):
#     try:
#         embedding_vector = model[word]
#         embedding_matrix[i] = embedding_vector
#     except KeyError:
#         continue
# del model , embed
from keras import layers
from keras import models
cols=["qt","qb","aw"]
usecols=["qtuse","qbuse","awuse"]
embed_dim=100
inputs = []
outputs = []
# for c in cols:
#     inp = layers.Input(shape=(max_len,), name=c)
#     out = layers.Embedding(len(word_index) + 1,
#                      300,
#                      weights=[embedding_matrix],
#                      input_length=max_len,
#                      trainable=False)(inp)
#     out = layers.SpatialDropout1D(0.5)(out)
# #     out = layers.Reshape(target_shape=(embed_dim, ))(out)
#     out=layers.Flatten()(out)
#     inputs.append(inp)
#     outputs.append(out)
def create_model():
  for i,c in ernumerate(usecols):
      inp=layers.Input(shape=(use_test[i].shape[1],),name=c)
      out=layers.Flatten()(inp)
      inputs.append(inp)
      outputs.append(out)

  x = layers.Concatenate()(outputs)
  # x=GRU(300, dropout=0.3, recurrent_dropout=0.3)(x)
  x = BatchNormalization()(x)
  x = Dropout(0.8)(x)
  x = Dense(100, activation="relu")(x)
  x = Dropout(0.8)(x)
  x = BatchNormalization()(x)
  x = Dense(100, activation="relu")(x)
  x = Dropout(0.8)(x)
  x = BatchNormalization()(x)
  y = Dense(30, activation="sigmoid")(x)

  model = models.Model(inputs=inputs, outputs=y)
  model.compile(loss='binary_crossentropy', optimizer='adam')
  return model
del embed
keras.backend.clear_session()
import datetime
import time
from keras import callbacks
import os
log_folder = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
# tb_callback = callbacks.TensorBoard(
#     log_dir=os.path.join('tb-logs', log_folder),
# )

# Best model callback
# bm_callback = callbacks.ModelCheckpoint(
#     filepath=os.path.join('tb-logs', log_folder, 'bm.h5'),
#     save_best_only=True,
#     save_weights_only=False
# )

earlystopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001,\
                              patience=5, verbose=0, mode='min')


rlr = callbacks.ReduceLROnPlateau( monitor='val_loss',\
                                  factor=0.1, patience=10, verbose=0, \
                                  cooldown=0, min_lr=1e-6)
mc= callbacks.ModelCheckpoint('best_model_use.h5', monitor='val_loss', mode='min', save_best_only=True)
CALLBACKS=[earlystopping,rlr,mc]
# SpearmanRhoCallback(training_data=({"qt":qttrain_pad,"qb":qbtrain_pad,"aw":awtrain_pad}, y_train), validation_data=({"qt":qtvalid_pad,"qb":qbvalid_pad,"aw":awvalid_pad},y_valid),
#                                        patience=5, model_name=u'best_model_batch.h5')
EPOCHS=1
BATCH_SIZE=1
CLASS_WEIGHTS=None
predictions = np.zeros((len(test),len(targets)))
for train_index, test_index in kf.split(train):
    print("TRAIN:", train_index, "TEST:", test_index)
    qt_train=train.loc[train_index,"question_title"].values
    qb_train=train.loc[train_index,'question_body'].values
    aw_train=train.loc[train_index,'answer'].values
    qt_valid=train.loc[test_index,"question_title"].values
    qb_valid=train.loc[test_index,'question_body'].values
    aw_valid=train.loc[test_index,'answer'].values
    
    y_train=train.loc[train_index,targets].values
    y_valid=train.loc[test_index,targets].values

    # qttrain_seq = token.texts_to_sequences(qt_train)
    # qtvalid_seq = token.texts_to_sequences(qt_valid)
    # qttest_seq = token.texts_to_sequences(qt_test)

    # qbtrain_seq = token.texts_to_sequences(qb_train)
    # qbvalid_seq = token.texts_to_sequences(qb_valid)
    # qbtest_seq = token.texts_to_sequences(qb_test)

    # awtrain_seq = token.texts_to_sequences(aw_train)
    # awvalid_seq = token.texts_to_sequences(aw_valid)
    # awtest_seq = token.texts_to_sequences(aw_test)

    # # # zero pad the sequences
    # qttrain_pad = sequence.pad_sequences(qttrain_seq, maxlen=max_len)
    # qtvalid_pad = sequence.pad_sequences(qtvalid_seq, maxlen=max_len)
    # qttest_pad = sequence.pad_sequences(qttest_seq, maxlen=max_len)

    # qbtrain_pad = sequence.pad_sequences(qbtrain_seq, maxlen=max_len)
    # qbvalid_pad = sequence.pad_sequences(qbvalid_seq, maxlen=max_len)
    # qbtest_pad = sequence.pad_sequences(qbtest_seq, maxlen=max_len)

    # awtrain_pad = sequence.pad_sequences(awtrain_seq, maxlen=max_len)
    # awvalid_pad = sequence.pad_sequences(awvalid_seq, maxlen=max_len)
    # awtest_pad = sequence.pad_sequences(awtest_seq, maxlen=max_len)

    embed = hub.load(module_url)
    qtusetrain=embed(qt_train)["outputs"].numpy()
    qbusetrain=embed(qb_train)["outputs"].numpy()
    awusetrain=embed(aw_train)["outputs"].numpy()
    qtusevalid=embed(qt_valid)["outputs"].numpy()
    qbusevalid=embed(qb_valid)["outputs"].numpy()
    awusevalid=embed(aw_valid)["outputs"].numpy()
    del embed
    keras.backend.clear_session()
    model=create_model()
    history= model.fit(
        x={"qtuse":qtusetrain,"qbuse":qbusetrain,"awuse":awusetrain},
        y=y_train,
        validation_data=({"qtuse":qtusevalid,"qbuse":qbusevalid,"awuse":awusevalid},y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=CLASS_WEIGHTS,
        callbacks=CALLBACKS,
        verbose=2)
    temp_preds=model.predict({"qtuse":qtusetest,"qbuse":qbusetest,"awuse":awusetest})
    predictions+=temp_preds
    keras.backend.clear_session()
    break

# preds=model.predict({"qt":qttest_pad,"qb":qbtest_pad,"aw":awtest_pad})
preds=predictions#/NFOLDS
submission=pd.DataFrame(preds,columns=targets)
submission[ids[0]]=test[ids[0]]
submission.to_csv("submission.csv", index = False)
submission.head()