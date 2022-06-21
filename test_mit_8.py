import numpy as np
import os
import wfdb
from collections import Counter
import pickle
import random
import sys
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
import warnings

import keras
from keras.models import Sequential
from keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout
from keras.layers import Input
from keras.models import Model
from tensorflow.keras.layers import BatchNormalization
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt


def valt(mp):
    return (len(mp),mp[10],mp[11],mp[12])

def drop_nan(data):
    j = 0
    for x in data:
        if pd.isnull(x):
            data[j] = -1
        j = j + 1


def network(X_train,y_train,X_test,y_test):
    

    im_shape=(X_train.shape[1],1)
    inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')
    conv1_1=Convolution1D(128,(3), activation='relu', input_shape=im_shape)(inputs_cnn)
    pool1=MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
    conv2_1=Convolution1D(128, (3), activation='relu', input_shape=im_shape)(pool1)
    pool2=MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv2_1)

    flatten=Flatten()(pool1)
    dense_end1 = Dense(128, activation='relu')(flatten)
    main_output = Dense(9, activation='softmax', name='main_output')(dense_end1)
    
    
    model = Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=8),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    history=model.fit(X_train, y_train,epochs=1,callbacks=callbacks, batch_size=32,validation_data=(X_test,y_test))
    model.load_weights('best_model.h5')
    return(model,history)



def evaluate_model(history,X_test,y_test,model):
    scores = model.evaluate((X_test),y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    print(history)
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    
    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
    #target_names=['0','1','2','3','4']
    
    y_true=[]
    for element in y_test:
        y_true.append(np.argmax(element))
    prediction_proba=model.predict(X_test)
    prediction=np.argmax(prediction_proba,axis=1)
    cnf_matrix = confusion_matrix(y_true, prediction)



if __name__ == "__main__":

    rhy_dict = {'Sinus rhythm' : 1. ,
                'Sinus tachycardia' : 2. ,
                'Sinus bradycardia' : 3. ,
                'Sinus arrhythmia' : 4. ,
                'Irregular sinus rhythm' : 5. ,
                'Atrial fibrillation' : 6. ,
                'Atrial flutter, typical' : 7. ,
                'Abnormal rhythm' : 8. }

    elec_dict = {'Electric axis of the heart: normal' : 1. ,
                'Electric axis of the heart: left axis deviation' : 2. ,
                'Electric axis of the heart: vertical' : 3. ,
                'Electric axis of the heart: horizontal' : 4. ,
                'Electric axis of the heart: right axis deviation' : 5. ,
                -1. : -1. }

    con_dict = {'Sinoatrial blockade, undefined' : 1. ,
                'I degree av block' : 2. ,
                'Iii degree av-block' : 3. ,
                'Incomplete right bundle branch block' : 4. ,
                'Incomplete left bundle branch block' : 5. ,
                'Left anterior hemiblock' : 6. ,
                'Complete right bundle branch block' : 7. ,
                'Complete left bundle branch block' : 8. ,
                'Non-specific intravintricular conduction delay' : 9. ,
                -1. : -1. }


    ext_dict = {'Atrial extrasystole: undefined' : 1. ,
                'Atrial extrasystole: low atrial' : 2. ,
                'Atrial extrasystole: left atrial' : 3. ,
                'Atrial extrasystole, SA-nodal extrasystole' : 4. ,
                'Atrial extrasystole, type: single pac' : 5. ,
                'Atrial extrasystole, type: bigemini' : 6. ,
                'Atrial extrasystole, type: quadrigemini' : 7. ,
                'Atrial extrasystole, type: allorhythmic pattern' : 8. ,
                'Ventricular extrasystole, morphology: polymorphic' : 9. ,
                'Ventricular extrasystole, localisation: rvot, anterior wall' : 10. ,
                'Ventricular extrasystole, localisation: rvot, antero-septal part' : 11. ,
                'Ventricular extrasystole, localisation: IVS, middle part' : 12. ,
                'Ventricular extrasystole, localisation: LVOT, LVS' : 13. ,
                'Ventricular extrasystole, localisation: LV, undefined' : 14. ,
                'Ventricular extrasystole, type: single pvc' : 15. ,
                'Ventricular extrasystole, type: intercalary pvc' : 16. ,
                'Ventricular extrasystole, type: couplet' : 17. ,
                -1. : -1. }


    hyp_dict = {'Right atrial hypertrophy' : 1. ,
                'Left atrial hypertrophy' : 2. ,
                'Right atrial overload' : 3. ,
                'Left atrial overload' : 4. ,
                'Left ventricular hypertrophy' : 5. ,
                'Right ventricular hypertrophy' : 6. ,
                'Left ventricular overload' : 7. ,
                -1. : -1. }


    card_dict = {'Pacemaker presence, undefined' : 1. ,
                'P-synchrony' : 2. ,
                -1. : -1. }


    isch_dict = {'Stemi: anterior wall' : 1. ,
                'Stemi: lateral wall' : 2. ,
                'Stemi: septal' : 3. ,
                'Stemi: inferior wall' : 4. ,
                'Stemi: apical' : 5. ,
                'Ischemia: anterior wall' : 6. ,
                'Ischemia: lateral wall' : 7. ,
                'Ischemia: septal' : 8. ,
                'Ischemia: inferior wall' : 9. ,
                'Ischemia: posterior wall' : 10. ,
                'Ischemia: apical' : 11. ,
                'Scar formation: lateral wall' : 12. ,
                'Scar formation: septal' : 13. ,
                'Scar formation: inferior wall' : 14. ,
                'Scar formation: posterior wall' : 15. ,
                'Scar formation: apical' : 16. ,
                'Undefined ischemia/scar/supp.NSTEMI: anterior wall' : 17. ,
                'Undefined ischemia/scar/supp.nstemi: lateral wall' : 18. ,
                'Undefined ischemia/scar/supp.NSTEMI: septal' : 19. ,
                'Undefined ischemia/scar/supp.nstemi: inferior wall' : 20. ,
                'Undefined ischemia/scar/supp.nstemi: posterior wall' : 21. ,
                'Undefined ischemia/scar/supp.nstemi: apical' : 22. ,
                -1. : -1. }


    nons_dict = {'Non-specific repolarization abnormalities: inferior wall' : 1. ,
                'Non-specific repolarization abnormalities: lateral wall' : 2. ,
                'Non-specific repolarization abnormalities: anterior wall' : 3. ,
                'Non-specific repolarization abnormalities: posterior wall' : 4. ,
                'Non-specific repolarization abnormalities: apical' : 5. ,
                'Non-specific repolarization abnormalities: septal' : 6. ,
                -1. : -1. }


    oth_dict = {'Early repolarization syndrome' : 1. ,
                -1. : -1. }

    sex_dict = {'M' : 1. ,
                'F' : 0. }





    path = 'ludb'
    save_path = ''
    # valid_lead = ['MLII', 'II', 'I', 'MLI', 'V5'] 
    #valid_lead = ['MLII'] 
    fs_out = 500
    test_ratio = 0.2

    with open(os.path.join(path, 'RECORDS'), 'r') as fin:
        all_record_name = fin.read().strip().split('\n')

    #all_record_name

    data_files = ["ludb/data/" + file for file in os.listdir("ludb/data/") if ".dat" in file]


    data_files = sorted(data_files, key=valt)

    #data_files

    chanel = []
    signal_out = []
    signal_all = []


    for miki in data_files:
        signal_out = []
        for i in range(12):
            chanel = []
            chanel.append(i)
            sig = wfdb.rdsamp(miki[:-4], channels=chanel)
            sig_1d = np.ravel(sig[0])
            signal_out.extend(sig_1d)

        signal_all.append(signal_out)

    df = pd.DataFrame(signal_all)

    df_2 = pd.read_csv(path+'/ludb.csv')


    df_2['Age'] = df_2['Age'].str.split('\n').str[0]
    df_2['Sex'] = df_2['Sex'].str.split('\n').str[0]
    df_2['Ischemia'] = df_2['Ischemia'].str.split('\n').str[0]
    df_2['Cardiac pacing'] = df_2['Cardiac pacing'].str.split('\n').str[0]
    df_2['Extrasystolies'] = df_2['Extrasystolies'].str.split('\n').str[0]
    df_2['Non-specific repolarization abnormalities'] = df_2['Non-specific repolarization abnormalities'].str.split('\n').str[0]
    df_2['Hypertrophies'] = df_2['Hypertrophies'].str.split('\n').str[0]
    df_2['Electric axis of the heart'] = df_2['Electric axis of the heart'].str.split('\n').str[0]
    df_2['Rhythms'] = df_2['Rhythms'].str.split('\n').str[0]
    df_2['Conduction abnormalities'] = df_2['Conduction abnormalities'].str.split('\n').str[0]
    df_2['Other states'] = df_2['Other states'].str.split('\n').str[0]



    drop_nan(df_2['Conduction abnormalities'])
    drop_nan(df_2['Extrasystolies'])
    drop_nan(df_2['Hypertrophies'])
    drop_nan(df_2['Cardiac pacing'])
    drop_nan(df_2['Ischemia'])
    drop_nan(df_2['Non-specific repolarization abnormalities'])
    drop_nan(df_2['Other states'])
    drop_nan(df_2['Electric axis of the heart'])



    df_2['Rhythms'] = [rhy_dict[item] for item in df_2['Rhythms']]
    df_2['Electric axis of the heart'] = [elec_dict[item] for item in df_2['Electric axis of the heart']]
    df_2['Conduction abnormalities'] = [con_dict[item] for item in df_2['Conduction abnormalities']]
    df_2['Extrasystolies'] = [ext_dict[item] for item in df_2['Extrasystolies']]
    df_2['Hypertrophies'] = [hyp_dict[item] for item in df_2['Hypertrophies']]
    df_2['Cardiac pacing'] = [card_dict[item] for item in df_2['Cardiac pacing']]
    df_2['Ischemia'] = [isch_dict[item] for item in df_2['Ischemia']]
    df_2['Non-specific repolarization abnormalities'] = [nons_dict[item] for item in df_2['Non-specific repolarization abnormalities']]
    df_2['Other states'] = [oth_dict[item] for item in df_2['Other states']]
    df_2['Sex'] = [sex_dict[item] for item in df_2['Sex']]


    df_2['Age'][df_2['ID'] == 34] = '89'

    #df_2.astype('float')
    df_2['Age'] = df_2['Age'].astype('float')

    df_all = pd.concat([df,df_2], axis=1)


    all_record_new = []

    for code in all_record_name:
        all_record_new.append(int(code[5:]))


    predictors= df_all

    ps = ['Rhythms', 'Electric axis of the heart', 'Conduction abnormalities', 'Extrasystolies', 'Hypertrophies', 'Cardiac pacing', 'Ischemia', 'Non-specific repolarization abnormalities', 'Other states']
    target = df_all[ps] # Strength column
    predictors.drop(['Rhythms', 'Electric axis of the heart', 'Conduction abnormalities', 'Extrasystolies', 'Hypertrophies', 'Cardiac pacing', 'Ischemia', 'Non-specific repolarization abnormalities', 'Other states'],1,inplace=True)
    predictors.drop('ID',1,inplace=True)


    predictors['Age'] = (predictors['Age'] - predictors['Age'].mean()) / predictors['Age'].std()
    predictors['Sex'] = (predictors['Sex'] - predictors['Sex'].mean()) / predictors['Sex'].std()

    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.30, random_state=42)

    model,history=network(X_train,y_train,X_test,y_test)


