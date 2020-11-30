#!/usr/bin/python3

from pymongo import MongoClient
import mne
import pandas as pd
import json
import pickle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pywt
import smtplib

def sendMail():
    message = 'Se ejecuto un proceso de analisis, por favor validar en la aplicacion'
    subject = 'ANALISIS APLICACION - EEG'

    message = 'Subject: {}\n\n{}'.format(subject,message)

    server =  smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('ANALISISEEG@gmail.com','hospital7913')

    server.sendmail('ANALISISEEG@gmail.com', 'jf.zuluaga10@uniandes.edu.co', message)

    server.quit()


def getfeatures(filename):

    dataframe = []
    dataframe = preprocess_signal(filename)
    X_test = pd.DataFrame(WaveletTransform(dataframe)) 

    return X_test


def load_defaults():
    global lo_freq
    global hi_freq
    global chn_lst
    global filter_method
    global epoch_duration
    global threshold_uv
    global clean_initial
    global clean_tail
    global std_threshold

    lo_freq=0.5 # Hz
    hi_freq=32 # Hz
    filter_method='iir'
    epoch_duration = 1 # segundos
    threshold_uv = 600 # mV
    clean_initial = 300 # segundos
    clean_tail = 90 # segundos 
    std_threshold = 1 # mV
    
    chn_lst=['FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', \
             'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2']

def extract_edf_info(fname):
    global raw_edf
    global chn_idx
    global sf
    global chn_lst

    raw_edf = mne.io.read_raw_edf(fname)
    chn_idx = mne.pick_channels(raw_edf.info['ch_names'], include=chn_lst, exclude=[])
    sf = raw_edf.info['sfreq']
        
def extract_raw_data():
    global raw_data
    global chn_idx
    global raw_edf

    raw_data = raw_edf.get_data(picks=chn_idx)

def apply_bandpass_filter():
    global lo_freq
    global hi_freq
    global sf
    global filter_method
    global filtered_data
    global raw_data
    
    filtered_data = mne.filter.filter_data(raw_data, picks=None, sfreq=sf, l_freq=lo_freq, h_freq=hi_freq, \
                                    method=filter_method)

def estimate_num_epochs():
    global filtered_data
    global sf
    global num_epochs
    global epoch_duration
    global n_points
    
    _, n_points = filtered_data.shape
    eeg_duration = round(n_points/sf,0)
    num_epochs = int(round(eeg_duration/epoch_duration,0))

def calc_batch_size():
    global batch_size
    global sf
    global epoch_duration
    
    batch_size = int(sf*epoch_duration)
    
def create_tmp_dataframe():
    global chn_idx
    global filtered_data
    global eeg_df
    global chn

    chn = len(chn_idx)
    eeg_cols = []
    
    for i in range(0, chn):
        eeg_cols.append('ch_'+str(i))
    
    eeg_df = pd.DataFrame(np.transpose(filtered_data*1000000), columns=eeg_cols)

def mark_epochs_xtreme_mvolts():
    global num_epochs
    global batch_size
    global chn
    global threshold_uv
    global clean_initial
    global epoch_duration
    global filtered_data
    global remove_set
    
    s_epoch = 0
    remove_set = set()
    
    n_epochs_ci = int(clean_initial/epoch_duration)
    
    for epoch in range(0, num_epochs):
        e_epoch = s_epoch + batch_size
        
        for batch in range(0, chn):
            f_s = filtered_data[batch][s_epoch:e_epoch]*1000000 # x 1000000 para convertir a mV.
            if (np.max(f_s) > threshold_uv) \
            or (np.min(f_s) < (-1*threshold_uv))\
            or (epoch < n_epochs_ci):
            # or ((-1*std_threshold) < np.std(f_s) < std_threshold): # No borrar.
                remove_set.add(epoch)
        
        s_epoch = e_epoch

def define_clean_tail():
    global n_points
    global clean_tail
    global sf
    global s_clean_tail
    global e_clean_tail
    
    tail_size = clean_tail*sf
    s_clean_tail = int(n_points - tail_size)
    e_clean_tail = int(s_clean_tail + tail_size)
    
def prepare_removal_indices():
    global remove_set
    global remove_idx
    global batch_size
    global s_clean_tail
    global e_clean_tail

    remove_idx = []
    
    for r in remove_set:
        ri = int(r*batch_size)
        rf = int(ri+batch_size)
        for i in range(ri, rf):
            remove_idx.append(i)
            
    for i in range(s_clean_tail, e_clean_tail):
        remove_idx.append(i)
    
def remove_bad_epochs():
    global eeg_df
    global remove_idx
    
    eeg_df.drop(remove_idx, inplace=True)
    eeg_df.reset_index(drop=True, inplace=True)
    
def preprocess_signal(fname):
    global eeg_df
    
    load_defaults()
    extract_edf_info(fname)
    extract_raw_data()
    apply_bandpass_filter()
    estimate_num_epochs()
    calc_batch_size()
    create_tmp_dataframe()
    mark_epochs_xtreme_mvolts()
    define_clean_tail()
    prepare_removal_indices()
    remove_bad_epochs()
    
    return eeg_df.copy()

def WaveletTransform(dataTrain):
    
    cA,cD = pywt.dwt(dataTrain, 'db1')
    
    return cA


print ('************************ - INICIO PROCESO - ************************')

# PASO 1: Conexión al Server de MongoDB Pasandole el host y el puerto
mongoClient = MongoClient('localhost',27017)

# PASO 2: Conexión a la base de datos
db = mongoClient.taller_2

# PASO 3: Obtenemos una coleccion para trabajar con ella
collectionOne = db.LoadDataProces
collectionTwo = db.Prediction


cursor = collectionOne.find({"process" :'0'})

for loadDataProces in cursor:
    
        fileName = '/Users/juanzuluaga/Documents/ProyectoFinal/aplicacion/APPEEG/media/' + loadDataProces['eeg']

        filenameModel = '/Users/juanzuluaga/Documents/ProyectoFinal/aplicacion/APPEEG/media/KNeighborsClassifier.sav'
        loaded_model = pickle.load(open(filenameModel, 'rb'))

        X_test = getfeatures(fileName)

        y_predict = loaded_model.predict(X_test)

        sizeArray = len(y_predict)

        epi = 0

        for con in y_predict:
            if con == 0:
                epi = epi + 1

        prediction = epi / sizeArray

        print('***************************************************************')
        print("Prediction = " + str(prediction))
        print('***************************************************************')

        if prediction > 0.4 :
            loadDataProces['prediction'] = '2'
        else:
            loadDataProces['prediction'] = '0'

        loadDataProces['process'] = '1'

        collectionTwo.insert(loadDataProces)
    
        collectionOne.find_one_and_update({"_id" : loadDataProces['_id']},
        {"$set":
            {"process": "1"}
        },upsert=True
        )

        sendMail()
   


mongoClient.close()
print ('************************ - FIN PROCESO - ************************')