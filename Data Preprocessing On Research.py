# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 13:07:34 2018

@author: MAYANK
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('KDDTrain+.csv')



#dos attack
x_dos_back=dataset.loc[dataset['attack_type'] == 'back']
x_dos_land=dataset.loc[dataset['attack_type'] == 'land']
x_dos_neptune=dataset.loc[dataset['attack_type'] == 'neptune']
x_dos_pod=dataset.loc[dataset['attack_type'] == 'pod']
x_dos_smurf=dataset.loc[dataset['attack_type'] == 'smurf']
x_dos_teardrop=dataset.loc[dataset['attack_type'] == 'teardrop']
#probe
x_probe_ipsweep=dataset.loc[dataset['attack_type'] == 'ipsweep']
x_probe_nmap=dataset.loc[dataset['attack_type'] == 'nmap']
x_probe_portsweep=dataset.loc[dataset['attack_type'] == 'portsweep']
x_probe_satan=dataset.loc[dataset['attack_type'] == 'satan']
#r2l
x_r2l_ftp_write=dataset.loc[dataset['attack_type'] == 'ftp_write']
x_r2l_guess_passwd=dataset.loc[dataset['attack_type'] == 'guess_passwd']
x_r2l_imap=dataset.loc[dataset['attack_type'] == 'imap']
x_r2l_multihop=dataset.loc[dataset['attack_type'] == 'multihop']
x_r2l_phf=dataset.loc[dataset['attack_type'] == 'phf']
x_r2l_spy=dataset.loc[dataset['attack_type'] == 'spy']
x_r2l_warezclient=dataset.loc[dataset['attack_type'] == 'warezclient']
x_r2l_warezmaster=dataset.loc[dataset['attack_type'] == 'warezmaster']
#u2r
x_u2r_buffer_overflow=dataset.loc[dataset['attack_type'] == 'buffer_overflow']
x_u2r_loadmodule=dataset.loc[dataset['attack_type'] == 'loadmodule']
x_u2r_perl=dataset.loc[dataset['attack_type'] == 'perl']
x_u2r_rootkit=dataset.loc[dataset['attack_type'] == 'rootkit']
#normal
x_normal=dataset.loc[dataset['attack_type'] == 'normal']


#random subset samples

#if they are numpy objects then concat procedure is different
#random_subset_x_dos_neptune=x_dos_neptune[np.random.choice(x_dos_neptune.shape[0], 15287, replace=False), :]
#x_dos_neptune is pandas object so randomization method is different
random_subset_x_dos_neptune=x_dos_neptune.sample(n=15287)
#x_normal is pandas object so randomization method is different
random_subset_x_normal = x_normal.sample(n=8000)




#concatenate all dataset
#if they are numpy objects then concat procedure is different
#x_u2r=np.concatenate((x_u2r_buffer_overflow,x_u2r_loadmodule,x_u2r_perl, x_u2r_rootkit))
#x_r2l=np.concatenate((x_r2l_ftp_write,x_r2l_guess_passwd,x_r2l_imap,x_r2l_multihop,x_r2l_phf,x_r2l_spy,x_r2l_warezclient,x_r2l_warezmaster))
#x_probe=np.concatenate((x_probe_ipsweep,x_probe_nmap,x_probe_portsweep, x_probe_satan))
#x_dos=np.concatenate((x_dos_back,x_dos_land,random_subset_x_dos_neptune,x_dos_pod,x_dos_smurf,x_dos_teardrop))

x_u2r = pd.concat([x_u2r_buffer_overflow,x_u2r_loadmodule,x_u2r_perl, x_u2r_rootkit])
x_r2l=pd.concat([x_r2l_ftp_write,x_r2l_guess_passwd,x_r2l_imap,x_r2l_multihop,x_r2l_phf,x_r2l_spy,x_r2l_warezclient,x_r2l_warezmaster])
x_probe=pd.concat([x_probe_ipsweep,x_probe_nmap,x_probe_portsweep, x_probe_satan])
x_dos=pd.concat([x_dos_back,x_dos_land,random_subset_x_dos_neptune,x_dos_pod,x_dos_smurf,x_dos_teardrop])







#merge all dataset to get my custom dataset

x = pd.concat([x_u2r,x_r2l,x_probe,x_dos,random_subset_x_normal])

#label the attack_type class
x['attack_type'] = x['attack_type'].map({'normal': 1, 'back': 2, 'land': 2, 'neptune': 2, 'pod': 2, 'smurf': 2, 'teardrop': 2, 'ipsweep': 3, 'nmap': 3, 'portsweep': 3, 'satan': 3, 'ftp_write': 4, 'guess_passwd': 4, 'imap': 4, 'multihop': 4, 'phf': 4, 'spy': 4, 'warezclient': 4, 'warezmaster': 4, 'buffer_overflow': 5, 'loadmodule': 5, 'perl': 5, 'rootkit': 5})


x = x.sample(frac=1)#randomization full set

x=x.iloc[:,0:42].values

dx=pd.DataFrame(x, columns=['duration',	'protocol_type'	,'service'	,'flag',	'src_bytes',	'dst_bytes',	'land',	'wrong_fragment',	'urgent',	'hot',	'num_failed_logins',	'logged_in',	'num_compromised',	'root_shell',	'su_attempted',	'num_root',	'num_file_creations	','num_shells',	'num_access_files',	'num_outbound_cmds',	'is_host_login',	'is_guest_login',	'count',	'srv_count',	'serror_rate',	'srv_serror_rate',	'rerror_rate',	'srv_rerror_rate',	'same_srv_rate	','diff_srv_rate',	'srv_diff_host_rate',	'dst_host_count',	'dst_host_srv_count',	'dst_host_same_srv_rate',	'dst_host_diff_srv_rate',	'dst_host_same_src_port_rate',	'dst_host_srv_diff_host_rate',	'dst_host_serror_rate',	'dst_host_srv_serror_rate',	'dst_host_rerror_rate',	'dst_host_srv_rerror_rate','attack_type'])





#-------------------Data Preprocessing--------------
#CATAGORICAL VALUES

#for protocol_type categorical variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_protocol_type=LabelEncoder()
dx.values[:, 1]=labelencoder_protocol_type.fit_transform(dx.values[:, 1])


#for service categorical variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_service=LabelEncoder()
dx.values[:, 2]=labelencoder_service.fit_transform(dx.values[:, 2])


#for flag categorical variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_flag=LabelEncoder()
dx.values[:, 3]=labelencoder_flag.fit_transform(dx.values[:, 3])

'''
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
#dx.values[:, 0]=sc_x.fit_transform(dx.values[:, 0])
dx[['duration','src_bytes','dst_bytes']] = sc_x.fit_transform(dx[['duration','src_bytes','dst_bytes']].as_matrix())

'''




#save my custom dataset in csv file

dx.to_csv("customDatasetTrain-40703_tuples.csv")
