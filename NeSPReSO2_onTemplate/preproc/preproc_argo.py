import numpy as np
import mat73
from scipy.signal import convolve2d
from sklearn.decomposition import PCA
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime, timedelta
import pickle
import sys
sys.path.append("/home/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/eoas_pyutils/")
from io_utils.coaps_io_data import get_aviso_by_date, get_sst_by_date, get_sss_by_date


def datenum_to_datetime(matlab_datenum):
    def single_datenum_to_datetime(single_datenum):
        days_from_year_0_to_year_1 = 366
        python_datetime = datetime.fromordinal(int(single_datenum) - days_from_year_0_to_year_1) + timedelta(days=single_datenum % 1)
        return python_datetime

    vfunc = np.vectorize(single_datenum_to_datetime)
    return vfunc(matlab_datenum)


# Load and preprocess the data, and return the inputs and labels pre-PCA
def preprocArgoNoSat(path = "/home/jmiranda/SubsurfaceFields/Data/ARGO_GoM_20220920.mat", max_depth=2000):
    """
    Args:
    - path (str): File path to the dataset.
    - input_params (dict): Parameters to include as input.
    - max_depth (int): Maximum depth for profiles.

    Returns:
    - tuple: Input values and raw profiles for temperature and salinity.
    """
    # Load the data
    data = mat73.loadmat(path)
    
    # Get valid mask and filter data
    valid_mask = get_valid_mask(data)
    TEMP, SAL, RHO, ADT, TIME, LAT, LON = filter_and_fill_data(data, valid_mask, max_depth)
    PRES = np.arange(0, max_depth+1)
    
    return TIME, LAT, LON, TEMP, SAL, ADT, RHO, PRES

def get_valid_mask(data):
    """Get mask of valid profiles based on missing values."""
    temp_mask = np.sum(np.isnan(data['TEMP']), axis=0) <= 5
    sal_mask = np.sum(np.isnan(data['SAL']), axis=0) <= 5
    rho_mask = np.sum(np.isnan(data['RHO']), axis=0) <= 5
    return temp_mask & sal_mask & rho_mask

def filter_and_fill_data(data, valid_mask, max_depth):
    """Filter data using the mask and fill missing values."""
    TEMP = data['TEMP'][:max_depth+1, valid_mask]
    SAL = data['SAL'][:max_depth+1, valid_mask]
    RHO = data['RHO'][:max_depth+1, valid_mask]
    SSH = data['ADT_loc'][valid_mask]
    TIME = datenum_to_datetime(data['TIME'][valid_mask])
    LAT = data['LAT'][valid_mask]
    LON = data['LON'][valid_mask]
    
    # Fill missing values using interpolation
    for i in range(TEMP.shape[1]):
        valid_temp_idx = np.where(~np.isnan(TEMP[:, i]))[0]
        TEMP[:, i] = np.interp(range(TEMP.shape[0]), valid_temp_idx, TEMP[valid_temp_idx, i])
        valid_sal_idx = np.where(~np.isnan(SAL[:, i]))[0]
        SAL[:, i] = np.interp(range(SAL.shape[0]), valid_sal_idx, SAL[valid_sal_idx, i])
        valid_rho_idx = np.where(~np.isnan(RHO[:, i]))[0]
        RHO[:, i] = np.interp(range(RHO.shape[0]), valid_rho_idx, RHO[valid_rho_idx, i])
        
    return TEMP, SAL, RHO, SSH, TIME, LAT, LON

def get_COAPS_ssh_sst(TIME, LAT, LON, max_ct = np.inf):
    aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
    sst_folder = "/unity/f1/ozavala/DATA/GOFFISH/SST/OISST"
    
    unique_dates = sorted(list(set(TIME)))
    sst_data = np.nan * np.ones(len(TIME))
    aviso_data = np.nan * np.ones(len(TIME))
    
    ct = -1
        
    for idx, c_date in enumerate(unique_dates):
        if ct < max_ct:
            ct += 1
            date_idx = np.array([date_obj == c_date for date_obj in TIME])
            coordinates = np.array([LAT[date_idx], LON[date_idx]]).T
            
            try:
                aviso_date, aviso_lats, aviso_lons = get_aviso_by_date(aviso_folder, c_date, bbox=(min(LAT), max(LAT), min(LON), max(LON)))
                interpolator_ssh = RegularGridInterpolator((aviso_lats, aviso_lons), aviso_date.adt.values, bounds_error=False, fill_value=None)
                aviso_data[date_idx] = interpolator_ssh(coordinates)
            except Exception as e:
                print(f"{ct} \tNo SSH on: ", c_date, "Error: ", str(e))
                continue

            try:
                sst_date, sst_lats, sst_lons = get_sst_by_date(sst_folder, c_date, bbox=(min(LAT), max(LAT), min(LON), max(LON)))
                interpolator_sst = RegularGridInterpolator((sst_lats, sst_lons), sst_date.analysed_sst.values[0], bounds_error=False, fill_value=None)
                sst_data[date_idx] = interpolator_sst(coordinates)
            except Exception as e:
                print(f"{ct} \tNo SST on: ", c_date, "Error: ", str(e))
                continue

            # Check if data was actually filled
            if np.isnan(aviso_data[date_idx]).all():
                print(f"{ct} \tNo SSH on: {c_date}")
            if np.isnan(sst_data[date_idx]).all():
                print(f"{ct} \tNo SST on: {c_date}")
                
            
    aviso_data = np.array(aviso_data)
    sst_data = np.array(sst_data)

    return aviso_data, sst_data


def applyPCA(matrix, n_components=15):
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(matrix.T).T
    return pcs, pca

def inverse_PCA(pca, pcs):
        profiles = pca.inverse_transform(pcs).T
        return profiles
    
def load_data(inputs = None, outputs = None, ncomps = 15, max_depth = 2000):
    if inputs is None:
        inputs = {
            "time": True,
            "geo": True,
            "sst": True,
            "sss": True,
            "adt": True,
            "satSSH": False,
            "satSST": False,
            "depth": False
        }
    inputs = inputs
        
    if outputs is None:
        outputs = {
            "temp": True,
            "sal": True,
            "rho": True,
            "adt": True
        }
    outputs = outputs
    
    
    TIME, LAT, LON, TEMP, SAL, ADT, RHO, PRES = preprocArgoNoSat()
    satSSH, satSST = get_COAPS_ssh_sst(TIME, LAT, LON)
        
    T_pcs, T_pca = applyPCA(TEMP, ncomps)
    S_pcs, S_pca = applyPCA(SAL, ncomps)
    R_pcs, R_pca = applyPCA(RHO, ncomps)
    
    input_array = []
    
    if inputs["time"]:
        cosT = np.cos(2*np.pi*TIME%365)/365
        sinT = np.sin(2*np.pi*TIME%365)/365
        avgTc = np.mean(cosT)
        avgTs = np.mean(sinT)
        input_array.append(cosT)
        input_array.append(sinT)
        
    if inputs["geo"]:
        cosLa = np.cos(2*np.pi*LAT/180)
        sinLa= np.sin(2*np.pi*LAT/180)
        cosLo = np.cos(2*np.pi*LON/360)
        sinLo = np.sin(2*np.pi*LON/360)
        avgLac = np.mean(cosLa)
        avgLas = np.mean(sinLa)
        avgLos = np.mean(sinLo)
        avgLoc = np.mean(cosLo)
        
        input_array.append(cosLa)
        input_array.append(sinLa)
        input_array.append(cosLo)
        input_array.append(sinLo)
        
    if inputs["sst"]:
        input_array.append(TEMP[0,:])
        avgST = np.mean(TEMP[0,:])
    elif inputs["satSST"]:
        input_array.append(satSST)
    if inputs["sss"]:
        input_array.append(SAL[0,:])
    if inputs["ssh"]:
        input_array.append(ADT)
    elif inputs["satSSH"]:
        input_array.append(satSSH)
    if inputs["depth"]:
        input_array.append(PRES)
        
    
        

def __main__():
    TIME, LAT, LON, TEMP, SAL, ADT, RHO, PRES = preprocArgoNoSat()
    satSSH, satSST = get_COAPS_ssh_sst(TIME, LAT, LON)
    T_pcs, T_pca = applyPCA(TEMP)
    S_pcs, S_pca = applyPCA(SAL)
    R_pcs, R_pca = applyPCA(RHO)

    data_to_save = {
        'TIME': TIME,
        'LAT': LAT,
        'LON': LON,
        'TEMP': TEMP,
        'SAL': SAL,
        'ADT': ADT,
        'RHO': RHO,
        'PRES': PRES,
        'satSSH': satSSH,
        'satSST': satSST,
    }

    with open("ARGO_SAT_data.pkl", "wb") as file:
        pickle.dump(data_to_save, file)
    
#run main if main
if __name__ == "__main__":
    __main__()    




