
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential, Model
from keras.layers import Input, Dense, concatenate
#from keras.utils import np_utils
from keras import utils   
from sklearn.utils import shuffle
import numpy as np
from datetime import datetime
import os
import pickle
from sklearn.model_selection import KFold
import tensorflow as tf
import shap
#from shap import shap_tabular
from tensorflow.keras.models import load_model
import keras
from datetime import datetime
import argparse
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pickle
import csv
import random
import shap
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, precision_score, recall_score
import sys



class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
  """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """
  #
  def __init__(self, patience=0):
    super(EarlyStoppingAtMinLoss, self).__init__()
    self.patience = patience
    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None
  #
  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best as infinity.
    self.best = np.Inf
  #
  def on_epoch_end(self, epoch, logs=None):
    current = logs.get("loss")
    if current < 1E-04: 
        self.stopped_epoch = epoch
        self.model.stop_training = True
        print("Loss is smaller then 1E-06, stopping training.") 
    if np.less(current, self.best): #loss is decreasing
      self.best = current
      self.wait = 0
      # Record the best weights if current results is better (less).
      self.best_weights = self.model.get_weights()
    else:
      print("val_accuracy:"+str(logs.get('accuracy')))
      if logs.get('accuracy')> 0.99: #loss didn't change and training accuracy is 1
        self.stopped_epoch = epoch
        self.model.stop_training = True
        print("Training accuracy is 100%, stopping training.")
      else:#loss didn't change and training accuracy is less then 1 -> wait until patience
        self.wait += 1
        if self.wait >= self.patience:
          self.stopped_epoch = epoch
          self.model.stop_training = True
          print("Restoring model weights from the end of the best epoch.")
          self.model.set_weights(self.best_weights)



def pickle_data(fileN_p, dt_p):
    file=open(fileN_p,'wb')
    pickle.dump(dt_p, file)
    file.close()



def make_original_model(x_shape_p, y_shape_p, Nnodes_p, Nlayers, optimizer_p):
    model = Sequential()
    model.add(Dense(Nnodes_p, input_dim=x_shape_p, activation='relu', name ="original_model_layer_0", trainable = True))#, kernel_initializer=keras.initializers.GlorotUniform(seed=629)))
    for li in range(1, Nlayers+1):     
        model.add(Dense(Nnodes_p, activation='relu', name ="original_model_layer_" + str(li+1), trainable = True))#, kernel_initializer=keras.initializers.GlorotUniform(seed=629)))
    #
    model.add(Dense(y_shape_p, activation='softmax'))  # Output layer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_p, metrics=['accuracy'])
    return model



def make_level2_model(npath_p, y_shape_p, Nnodes_p, Nlayers, optimizer_p):
    model = Sequential()
    model.add(Dense(Nnodes_p, input_dim=npath_p, activation='relu', name ="level2_model_layer_0", trainable = True))
    for li in range(1, Nlayers+1):     
        model.add(Dense(Nnodes_p, activation='relu', name ="level2_model_layer_" + str(li+1), trainable = True))
    model.add(Dense(y_shape_p, activation='softmax'))  # Output layer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_p, metrics=['accuracy'])
    return model


# read pathways index
path_index=pd.read_csv(path_index_fileN, header=None, index_col=False)
#path_index=path_index.iloc[:,1]
path_index = path_index.values.ravel().tolist()



pathways = pd.read_csv(pathwayN, header=0, index_col=0) #747
path_names=np.array(pathways.columns)[path_index]

#data = pd.read_csv(dataN)

# number of pathways
npath = len(path_index)


############### pathway

time_stamp_f = open(time_log_file, 'w', encoding="utf-8")


with open(runN+'/data/y_train_cvdi'+str(3)+'_asTraining', 'rb') as file:
    y_train_p = pickle.load(file)

with open(runN+'/data/X_train_cvdi'+str(3)+'_asTraining', 'rb') as file:
    X_train_p = pickle.load(file)

with open(runN+'/data/y_train_cvdi'+str(4)+'_asTraining', 'rb') as file:
    y_test_p = pickle.load(file)

with open(runN+'/data/X_train_cvdi'+str(4)+'_asTraining', 'rb') as file:
    X_test_p = pickle.load(file)


########## original 

# Build neural network

outter_cv_train_pathway_predictions = []
outter_cv_test_pathway_predictions = []

for pi in path_index:
    with open( runN + 
            data_path1+str(pi), 'rb') as file:#this is predected on 0
        outter_cv_test_pathway_predictions.append(pickle.load(file))
    with open(runN + 
            data_path2+str(pi), 'rb') as file:
        outter_cv_train_pathway_predictions.append(pickle.load(file))

outter_cv_train_pathway_predictions = np.array(outter_cv_train_pathway_predictions)
outter_cv_train_pathway_predictions = outter_cv_train_pathway_predictions[:,:,1].T
#
outter_cv_test_pathway_predictions = np.array(outter_cv_test_pathway_predictions)
outter_cv_test_pathway_predictions = outter_cv_test_pathway_predictions[:,:,1].T

train_combined_x=np.hstack((outter_cv_train_pathway_predictions, X_train_p))
test_combined_x=np.hstack((outter_cv_test_pathway_predictions, X_test_p))

del outter_cv_test_pathway_predictions, X_train_p, outter_cv_train_pathway_predictions, X_test_p


model2_1 = make_level2_model(train_combined_x.shape[1], y_train_p.shape[1], Nnodes, Nlayers, optimizer)

# Train the model
model2_1.fit(train_combined_x, y_train_p, epochs=epoch_p, batch_size=batch_size_p, verbose=1, callbacks=[EarlyStoppingAtMinLoss(patience=patience_p)], validation_data=(test_combined_x, y_test_p))
# Evaluate the model
loss, accuracy = model2_1.evaluate(test_combined_x, y_test_p, verbose=0)
print(f'########## Original Plus Pathways Accuracy in the {3}th outter cv=-vold: {accuracy*100:.2f}%')


y_pred = model2_1.predict(test_combined_x)
y_pred = np.argmax(y_pred, axis=1)
y_test_bi = np.argmax(y_test_p, axis=1) 

cm = confusion_matrix(y_test_bi, y_pred)
sensitivity = recall_score(y_test_bi, y_pred, average='binary')  # 'binary' if binary classification, 'macro' for multiclass
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TN / (TN + FP)
f1 = f1_score(y_test_bi, y_pred, average='binary')  # 'binary' for binary, 'macro' for multiclass
kappa = cohen_kappa_score(y_test_bi, y_pred)


with open(runN+"/original_plus_pathways_accuracy", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["accuracy", "sensitivity", "specificity", "f1", "kappa"]])
    writer.writerows([[accuracy, sensitivity, specificity, f1, kappa]])


gene_name=pathways.index.tolist()
del pathways


cvdi=4
gene_importance_all_pathway = []

explainer = shap.DeepExplainer(model2_1, train_combined_x)
del train_combined_x
shap_values = explainer.shap_values(test_combined_x)
sub_pathway_time_e2 = datetime.now().replace(microsecond=0) #started at Sep 7 8:15 pm
shap_values = np.array(shap_values)
shap_importance = pd.DataFrame(list(zip( path_names.tolist() + gene_name, np.mean(np.abs(shap_values[:,:,0]), axis=0))),
                            columns=['Feature', 'SHAP Importance'])
gene_importance = shap_importance.sort_values(by="SHAP Importance", ascending=False)
gene_importance.to_csv(runN+"/original_plus_pathways_model_gene_importance_cvdi"+str(cvdi), index=False)

gene_shap_time_e = datetime.now().replace(microsecond=0)


