
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
import cp437


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


parser = argparse.ArgumentParser()

# read params
parser.add_argument('--dataN',
                    type=str)
parser.add_argument('--pathwayN',
                        type=str)
parser.add_argument('--Nlayers', default=[2,3,4,5], 
                            type=int,
                            nargs='*',
                            help='number of nodes in the last layer')
parser.add_argument('--Nnodes', default=[32,64,128,256,512,1024,2048], 
                            type=int,
                            nargs='*',
                            help='number of nodes in the last layer')
parser.add_argument('--optimizer', default=['adam', 'sgd', 'rmsprop'], 
                        type=str,
                        nargs='*')
parser.add_argument('--cell_type',
                        type=str)
parser.add_argument('--epoch',
                        type=int)
parser.add_argument('--cv_count',
                        type=int)
parser.add_argument('--cv_in_cv_count',
                        type=int)
parser.add_argument('--importance_sampleC',
                        type=int)
parser.add_argument('--patience', default = 5,
                        type=int)
parser.add_argument('--batch_size', default = 10,
                        type=int)
parser.add_argument('--output_prefix',
                        type=str)
parser.add_argument('--path_index_fileN',
                            type=str,
                            help='indices of pathways')
parser.add_argument('--runN',
                        type=str)


args = parser.parse_args()

dataN = args.dataN
pathwayN = args.pathwayN
Nlayers = args.Nlayers
Nnodes = args.Nnodes
optimizer = args.optimizer
cell_type = args.cell_type
epoch_p = args.epoch
cv_count = args.cv_count
n_splits = args.cv_in_cv_count
sampleC_for_importance = args.importance_sampleC
output_prefix = args.output_prefix 
time_log_file = output_prefix + "_time_log.txt"
patience_p = args.patience
batch_size_p = args.batch_size
path_index_fileN=args.path_index_fileN
runN = args.runN

############### This should be updated
Nlayers = Nlayers[0]
Nnodes = Nnodes[0]
optimizer = optimizer[0]



print("Arguments:")
print("Data: "+dataN)
print("Pathway: "+pathwayN)
print(f"Nlayers: {Nlayers}")
print(f"Nnodes: {Nnodes}")
print("Optimizer: "+optimizer)
print("Cell type: "+cell_type)
print(f"epoch_p: {epoch_p}")
print(f"cv_count: {cv_count}")
print(f"n_splits: {n_splits}")
print(f"sampleC_for_importance: {sampleC_for_importance}")
print(f"patience: {patience_p}")
print("Time log file:"+time_log_file)
print('\n')



def make_level2_model(npath_p, y_shape_p, Nnodes_p, Nlayers, optimizer_p):
    model = Sequential()
    model.add(Dense(Nnodes_p, input_dim=npath_p, activation='relu', name ="level2_model_layer_0", trainable = True))
    for li in range(1, Nlayers+1):     
        model.add(Dense(Nnodes_p, activation='relu', name ="level2_model_layer_" + str(li+1), trainable = True))
    model.add(Dense(y_shape_p, activation='softmax'))  # Output layer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_p, metrics=['accuracy'])
    return model



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




def stepwise_forward(target_pathways_fixed_p, target_pathways_testing_p,  y_train_p, y_test_p, Nnodes_p, Nlayers_p, optimizer_p, runN, epoch_p, batch_size_p, patience_p):
    #
    accuracies = []
    for test_pi in range(0, len(target_pathways_testing_p)):
        #
        model2_0 = make_level2_model(len(target_pathways_fixed_p)+1, y_train_p.shape[1], Nnodes_p, Nlayers_p, optimizer_p)
        #
        inner_cv_train_pathway_predictions = []
        inner_cv_test_pathway_predictions = []
        #
        for pi in target_pathways_fixed_p + [target_pathways_testing_p[test_pi]]:
            with open( runN + 
                    "/outter_cv_train_pathway_predictions/outter_cv_train_pathway_predictions_cvdi"+str(2)+"_pi"+str(pi), 'rb') as file:
                inner_cv_train_pathway_predictions.append(pickle.load(file))
            with open(runN + 
                    "/outter_cv_train_pathway_predictions/outter_cv_train_pathway_predictions_cvdi"+str(3)+"_pi"+str(pi), 'rb') as file:
                inner_cv_test_pathway_predictions.append(pickle.load(file))
        #
        inner_cv_train_pathway_predictions = np.array(inner_cv_train_pathway_predictions)
        inner_cv_train_pathway_predictions = inner_cv_train_pathway_predictions[:,:,1].T
        #
        inner_cv_test_pathway_predictions = np.array(inner_cv_test_pathway_predictions)
        inner_cv_test_pathway_predictions = inner_cv_test_pathway_predictions[:,:,1].T
        #
        model2_0.fit(inner_cv_train_pathway_predictions, y_train_p, epochs=epoch_p, batch_size=batch_size_p, verbose=1, callbacks=[EarlyStoppingAtMinLoss(patience=patience_p)], validation_data=(inner_cv_test_pathway_predictions, y_test_p))
        #
        loss, accuracy = model2_0.evaluate(inner_cv_test_pathway_predictions, y_test_p, verbose=0)
        accuracies.append(accuracy)
    return target_pathways_testing_p[accuracies.index(max(accuracies))], max(accuracies)



def make_original_model(x_shape_p, y_shape_p, Nnodes_p, Nlayers, optimizer_p):
    model = Sequential()
    model.add(Dense(Nnodes_p, input_dim=x_shape_p, activation='relu', name ="original_model_layer_0", trainable = True))#, kernel_initializer=keras.initializers.GlorotUniform(seed=629)))
    for li in range(1, Nlayers+1):     
        model.add(Dense(Nnodes_p, activation='relu', name ="original_model_layer_" + str(li+1), trainable = True))#, kernel_initializer=keras.initializers.GlorotUniform(seed=629)))
    #
    model.add(Dense(y_shape_p, activation='softmax'))  # Output layer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_p, metrics=['accuracy'])
    return model


# read pathways index
path_index=pd.read_csv(path_index_fileN, header=None, index_col=False, sep=" ")
#path_index=path_index.iloc[:,1]
path_index = path_index.values.ravel().tolist()



pathways = pd.read_csv(pathwayN, header=0, index_col=0) #747

data = pd.read_csv(dataN)

# number of pathways
npath = len(path_index)


###############

time_stamp_f = open(time_log_file, 'w', encoding="utf-8")


with open(runN+'/data/y_train_cvdi'+str(1)+'_asTraining', 'rb') as file:
    y_train_p = pickle.load(file)

with open(runN+'/data/y_train_cvdi'+str(2)+'_asTraining', 'rb') as file:
    y_train_p = np.concatenate((y_train_p, pickle.load(file)), axis=0)

with open(runN+'/data/y_train_cvdi'+str(3)+'_asTraining', 'rb') as file:
    y_train_p = np.concatenate((y_train_p, pickle.load(file)), axis=0)


with open(runN+'/data/X_train_cvdi'+str(1)+'_asTraining', 'rb') as file:
    X_train_p = pickle.load(file)

with open(runN+'/data/X_train_cvdi'+str(2)+'_asTraining', 'rb') as file:
    X_train_p = np.concatenate((X_train_p, pickle.load(file)), axis=0)

with open(runN+'/data/X_train_cvdi'+str(3)+'_asTraining', 'rb') as file:
    X_train_p = np.concatenate((X_train_p, pickle.load(file)), axis=0)


with open(runN+'/data/y_train_cvdi'+str(4)+'_asTraining', 'rb') as file:
    y_test_p = pickle.load(file)


with open(runN+'/data/X_train_cvdi'+str(4)+'_asTraining', 'rb') as file:
    X_test_p = pickle.load(file)



original_accuracy = [] # size: cv_count
dataset_level2_accuracy = [] # X_train based pathway accuracy size: cv_count 
dataset_pathway_accuracy = [] # X_test based pathway accuracy size: cv_count x pathway
dataset_gene_importance = [] #size: cv_count x sampleC_for_importance
dataset_pathway_importance = [] #size: cv_count x npath
outter_cv_pathway_accuracy_ave = [] # X_train based pathway accuracy averaged over k folds size: cv_count x pathway
outter_cv_level2_accuracy = [] # X_train based pathway accuracy averaged over k folds for each data set: cv_count

target_pathways = path_index # 0 corresponds to the first index in the index_file




########## outter cv
dataset_time_s =  datetime.now().replace(microsecond=0)



########## inner cv        
outter_cv_pathway_importance = [] # X_test based pathway importance size: sampleC_for_importance x pathway
outter_cv_gene_importance = [] # X_test based gene importance on importance_flag size: sampleC_for_importance x gene
outter_cv_pathway_accuracy = [] # X_test based gene importance on importance_flag size: sampleC_for_importance x gene
inner_cv_level2_accuracy = [] # X_train based level2 accuracy for inner cv:  n_splits
#
steps = 0



with open(runN+'/data/y_train_cvdi'+str(2)+'_asTraining', 'rb') as file:
    y_train = pickle.load(file)

with open(runN+'/data/y_train_cvdi'+str(3)+'_asTraining', 'rb') as file:
    y_val = pickle.load(file)

past_acc = [0]
fixed_pi=[target_pathways[0]]
testing_pi=target_pathways[1:]
for tpi in range(0,len(testing_pi)):
    if(past_acc[-1]>0.99):
        break
    inner_cv_time_s =  datetime.now().replace(microsecond=0)
    max_index, max_acc = stepwise_forward(target_pathways_fixed_p=fixed_pi, target_pathways_testing_p=testing_pi, y_train_p=y_train, y_test_p=y_val, Nnodes_p=Nnodes, Nlayers_p=Nlayers, optimizer_p=optimizer, runN=runN, epoch_p=1, batch_size_p=batch_size_p, patience_p=patience_p)
    inner_cv_time_e =  datetime.now().replace(microsecond=0)
    time_stamp_f.write(f'{tpi}th pathway stepwise forwarad calculation took {inner_cv_time_e - inner_cv_time_s}\n')
    if(max_acc < past_acc[-1]): 
        break
    past_acc.append(max_acc)
    fixed_pi.append(max_index)
    testing_pi.remove(max_index)

with open(runN+"/stepfoward_accuracy_cvdi"+str(3), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([past_acc])

with open(runN+"/stepfoward_final_path_index_cvdi"+str(3), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([fixed_pi])





with open(runN+'/data/y_train_cvdi'+str(3)+'_asTraining', 'rb') as file:
    y_train = pickle.load(file)

with open(runN+'/data/y_train_cvdi'+str(4)+'_asTraining', 'rb') as file:
    y_test = pickle.load(file)

outter_cv_train_pathway_predictions = []
outter_cv_test_pathway_predictions = []

for pi in fixed_pi:
    with open( runN + 
            "/outter_cv_train_pathway_predictions/outter_cv_train_pathway_predictions_cvdi4_pi"+str(pi), 'rb') as file:#this is predected on 0
        outter_cv_test_pathway_predictions.append(pickle.load(file))
    with open(runN + 
            "/outter_cv_train_pathway_predictions/outter_cv_train_pathway_predictions_cvdi3_pi"+str(pi), 'rb') as file:
        outter_cv_train_pathway_predictions.append(pickle.load(file))

outter_cv_train_pathway_predictions = np.array(outter_cv_train_pathway_predictions)
outter_cv_train_pathway_predictions = outter_cv_train_pathway_predictions[:,:,1].T
#
outter_cv_test_pathway_predictions = np.array(outter_cv_test_pathway_predictions)
outter_cv_test_pathway_predictions = outter_cv_test_pathway_predictions[:,:,1].T
#
model2_1 = make_level2_model(len(fixed_pi), y_train.shape[1], Nnodes, Nlayers, optimizer)
model2_1.fit(outter_cv_train_pathway_predictions, y_train, epochs=epoch_p, batch_size=batch_size_p, verbose=1, callbacks=[EarlyStoppingAtMinLoss(patience=patience_p)], validation_data=(outter_cv_test_pathway_predictions, y_test))
#
loss, accuracy = model2_1.evaluate(outter_cv_test_pathway_predictions, y_test, verbose=0)



y_pred = model2_1.predict(outter_cv_test_pathway_predictions)
y_pred = np.argmax(y_pred, axis=1)
y_test_bi = np.argmax(y_test, axis=1) 

cm = confusion_matrix(y_test_bi, y_pred)
sensitivity = recall_score(y_test_bi, y_pred, average='binary')  # 'binary' if binary classification, 'macro' for multiclass
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TN / (TN + FP)
f1 = f1_score(y_test_bi, y_pred, average='binary')  # 'binary' for binary, 'macro' for multiclass
kappa = cohen_kappa_score(y_test_bi, y_pred)


with open(runN+"/stepfoward_final_accuracy_cvdi"+str(4), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["accuracy", "sensitivity", "specificity", "f1", "kappa"]])
    writer.writerows([[accuracy, sensitivity, specificity, f1, kappa]])





########## gene shap
gene_shap_time_s =  datetime.now().replace(microsecond=0)
print(f'########## Calculating gene importance in the {cvdi}th outer cv-fold')

trained_models=[]
for tpi in fixed_pi:
    trained_models.append(tf.keras.models.load_model(runN+"/trained_models/outter_model_cvdi"+str(1)+"_pi"+str(tpi)+".keras"))



with open(runN+'/data/X_train_cvdi'+str(1)+'_asTraining', 'rb') as file:
    X_train = pickle.load(file)

with open(runN+'/data/X_train_cvdi'+str(4)+'_asTraining', 'rb') as file:
    X_test = pickle.load(file)


cvdi=4
gene_importance_all_pathway = []
for pathway_i in range(len(fixed_pi)):
    gene_importance = []
    pathways_sub=pathways.iloc[:,fixed_pi[pathway_i]]
    pathways_sub = pathways_sub[pathways_sub>0]
    #
    X_train_sub=X_train[:,np.where(pathways_sub>0)[0]]
    X_test_sub=X_test[:,np.where(pathways_sub>0)[0]]
    explainer = shap.DeepExplainer(trained_models[pathway_i], X_train_sub)
    shap_values = explainer.shap_values(X_test_sub)
    sub_pathway_time_e2 = datetime.now().replace(microsecond=0) #started at Sep 7 11:15 pm
    shap_values = np.array(shap_values)
    shap_importance = pd.DataFrame(list(zip(pathways_sub.index.tolist(), np.mean(np.abs(shap_values[:,:,0]), axis=0))),
                                columns=['Feature', 'SHAP Importance'])
    gene_importance = shap_importance.sort_values(by="SHAP Importance", ascending=False)
    gene_importance.to_csv(runN+"/gene_importance_cvdi"+str(cvdi)+"_pi"+str(fixed_pi[pathway_i]), index=False)


gene_shap_time_e = datetime.now().replace(microsecond=0)
time_stamp_f.write(f'Gene importance calculation the {cvdi}th outer cv-fold took {gene_shap_time_e - gene_shap_time_s}\n')




dataset_time_e =  datetime.now().replace(microsecond=0)
time_stamp_f.write(f'The entire process took {dataset_time_e - dataset_time_s}\n')

time_stamp_f.close()


