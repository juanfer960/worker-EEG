#!/usr/bin/python3

from pymongo import MongoClient
import mne
import pandas as pd
import json

def process():
    print("hola mundo worker")

print(process())

# PASO 1: Conexión al Server de MongoDB Pasandole el host y el puerto
mongoClient = MongoClient('localhost',27017)

# PASO 2: Conexión a la base de datos
db = mongoClient.taller_2

# PASO 3: Obtenemos una coleccion para trabajar con ella
collectionOne = db.LoadDataProces
collectionTwo = db.Prediction


cursor = collectionOne.find({"process" :'0'})

for loadDataProces in cursor:

    loadDataProces['prediction'] = '0'
    loadDataProces['process'] = '1'

    #cargarDatos

    edf_file = '/Users/juanzuluaga/Documents/ProyectoFinal/aplicacion/APPEEG/media/' + loadDataProces['eeg']
    data = mne.io.read_raw_edf(edf_file)
    raw_data = data.get_data()
    dataframe = pd.DataFrame.from_records(raw_data)

    name = str(loadDataProces['eeg'])
   
    min = 0
    max = 0
    cont = 0

    #cargar EDF BD
    for x in range(int(len(dataframe.columns)/10000)):
        max = max + 10000
        df1 = dataframe.iloc[:, min:max]
        df1['edf_name'] = loadDataProces['eeg']
        cont = cont + 1
        df1['row_count'] = cont 
        min = min + 10000

        records = json.loads(df1.T.to_json()).values()

        db.EDF.insert(records)
        df1.drop(columns=['edf_name', 'row_count'])
        
    
   
    # Actualizar tablas proceso
    collectionTwo.insert(loadDataProces)
   
    collectionOne.find_one_and_update({"_id" : loadDataProces['_id']},
    {"$set":
        {"process": "1"}
    },upsert=True
    )


mongoClient.close()
print ('PROCESO TERMINADO')


