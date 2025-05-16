from torchvision import datasets, transforms
import pickle
import numpy as np
import sys
import torch
sys.path.append("../base/")
from base_data_loader import BaseDataLoader



## % From old code (nespreso 1 on template)

# sys.path.append("../preproc/")
# from preproc_argo import preprocArgoNoSat, get_COAPS_ssh_sst, applyPCA, inverse_PCA

# class PCA_ArgoDataLoader(BaseDataLoader):
#     """
#     ARGO data loading
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=2, training=True, inputs = None, outputs = None, max_depth = 2000):
#         super().__init__(data_dir, batch_size, shuffle, validation_split, num_workers, training)

#         if data_dir.endswith('.pkl'):
#             with open(data_dir, "rb") as file:
#                 loaded_data = pickle.load(file)
#                 self.TIME = loaded_data['TIME']
#                 self.LAT = loaded_data['LAT']
#                 self.LON = loaded_data['LON']
#                 self.TEMP = loaded_data['TEMP']
#                 self.SAL = loaded_data['SAL']
#                 self.ADT = loaded_data['ADT']
#                 self.RHO = loaded_data['RHO']
#                 self.PRES = loaded_data['PRES']
#                 self.satSSH = loaded_data['satSSH']
#                 self.satSST = loaded_data['satSST']
#         elif data_dir.endswith('.mat'):
#             self.TIME, self.LAT, self.LON, self.TEMP, self.SAL, self.ADT, self.RHO, self.PRES = preprocArgoNoSat(data_dir)
#             self.satSSH, self.satSST = get_COAPS_ssh_sst(data_dir)
#         else:
#             raise ValueError("Provided data_dir does not match supported file types (.pickle or .mat)")
        
#         if inputs is None:
#             inputs = {
#                 "time": True,
#                 "lat": True,
#                 "lon": True,
#                 "sst": True,
#                 "sss": True,
#                 "adt": True,
#                 "satSSH": False,
#                 "satSST": False,
#                 "depth": False
#             }
            
#         self.inputs = inputs
            
#         if outputs is None:
#             outputs = {
#                 "temp": True,
#                 "sal": True,
#                 "rho": True,
#                 "adt": False
#             }
            
#         self.outputs = outputs
                    
#         self.data_inputs, self.input_norm_params = self.create_inputs()
#         self.data_outputs, self.output_norm_params = self.create_outputs()
    
#     def normalize_data(data):
#         mean = np.mean(data, axis=0)
#         std = np.std(data, axis=0)
#         return (data - mean) / (std + 1e-12), mean, std
        
#     def create_inputs(self):
#         inputs = []
        
#         input_array = []
    
#         if inputs["time"]:
#             cosT = np.cos(2*np.pi*self.TIME%365)/365
#             sinT = np.sin(2*np.pi*self.TIME%365)/365
#             avgTc = np.mean(cosT)
#             avgTs = np.mean(sinT)
#             input_array.append(cosT)
#             input_array.append(sinT)
            
#         if inputs["geo"]:
#             cosLa = np.cos(2*np.pi*self.LAT/180)
#             sinLa= np.sin(2*np.pi*self.LAT/180)
#             cosLo = np.cos(2*np.pi*self.LON/360)
#             sinLo = np.sin(2*np.pi*self.LON/360)
#             avgLac = np.mean(cosLa)
#             avgLas = np.mean(sinLa)
#             avgLos = np.mean(sinLo)
#             avgLoc = np.mean(cosLo)
            
#             input_array.append(cosLa)
#             input_array.append(sinLa)
#             input_array.append(cosLo)
#             input_array.append(sinLo)
            
#         if inputs["sst"]:
#             input_array.append(self.TEMP[0,:])
#             avgST = np.mean(self.TEMP[0,:])
#         elif inputs["satSST"]:
#             input_array.append(self.satSST)
#         if inputs["sss"]:
#             input_array.append(self.SAL[0,:])
#         if inputs["ssh"]:
#             input_array.append(self.ADT)
#         elif inputs["satSSH"]:
#             input_array.append(self.satSSH)
#         if inputs["depth"]:
#             input_array.append(self.PRES)
        
#         inputs_data = np.array(inputs_data).T
        
#         normalized_inputs , self.mean, self.std = self.normalize_data(inputs_data)
        
#         return torch.tensor(normalized_inputs, dtype=torch.float32)
    
#     def create_outputs(self):
#         outputs = []
        
#         output_array = []
        
#         if outputs["temp"]:
#             output_array.append(self.TEMP)
#         if outputs["sal"]:
#             output_array.append(self.SAL)
#         if outputs["rho"]:
#             output_array.append(self.RHO)
#         if outputs["adt"]:
#             output_array.append(self.ADT)
        
#         outputs_data = np.array(outputs_data).T
        
#         normalized_outputs , self.mean, self.std = self.normalize_data(outputs_data)
        
#         return torch.tensor(normalized_outputs, dtype=torch.float32)


# # normalizar os dados
#         # PCA self.Tpcs, self.Tpca = applyPCA(self.TEMP)
#         # PCA self.Spcs, self.Spca = applyPCA(self.SAL) 
#         # PCA self.Rpcs, self.Rpca = applyPCA(self.RHO)
        
#         # inverse PCA self.TEMP = inverse_PCA(self.Tpca, self.Tpcs)