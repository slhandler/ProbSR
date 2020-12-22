TRAINING_DATA_PATH = '../data/dataset_16_17.csv'
CALIBRATION_DATA_PATH = '../data/dataset_17_18.csv'
TESTING_DATA_PATH = '../data/dataset_18_19.csv'

SCORING_FUNC = 'average_precision'
HYPER_PARAM_N_ITER = 15
TARGET_COLUMN = 'cat_rt'

PREDICTOR_COLUMNS = ['dllwave_flux', 'dwpt2m', 'fric_vel', 'gflux', 'lat_hf',
                     'sat_irbt', 'sens_hf', 'hrrr_dT', 'soilT_diff',
                     'sfcT_hrs_ab_frez', 'sfcT_hrs_bl_frez', 'sfc_rough', 'sfc_temp',
                     'swave_flux', 'temp2m', 'tmp2m_hrs_ab_frez', 'tmp2m_hrs_bl_frez',
                     'tot_cloud', 'vbd_flux', 'wind10m', 'urban', 'd_ground']

COLUMN_DTYPES = {'dllwave_flux': 'float16', 'dwpt2m': 'float16',
                 'fric_vel': 'float16', 'gflux': 'float16', 'high_cloud': 'float16',
                 'lat_hf': 'float16', 'low_cloud': 'float16', 'mid_cloud': 'float16',
                 'sat_irbt': 'float16', 'sens_hf': 'float16', 'sfcT_hrs_ab_frez': 'float16',
                 'sfcT_hrs_bl_frez': 'float16', 'sfc_rough': 'float16', 'sfc_temp': 'float16',
                 'swave_flux': 'float16', 'temp2m': 'float16', 'tmp2m_hrs_ab_frez': 'float16',
                 'tmp2m_hrs_bl_frez': 'float16', 'tot_cloud': 'float16', 'uplwav_flux': 'float16',
                 'vbd_flux': 'float16', 'vdd_flux': 'float16', 'wind10m': 'float16',
                 'date_marker': 'float16', 'urban': 'float16', 'rural': 'float16', 'd_ground': 'float16',
                 'd_rad_d': 'float16', 'd_rad_u': 'float16', 'hrrr_dT': 'float16',
                 'cat_rt': 'float64', 'lat': 'float16', 'lon': 'float16', 'date': 'object', 'obs_rt': 'float16',
                 'PRECIPRATE': 'float16', '1HRAD': 'float16', '3HRAD': 'float16', 'snow_cover': 'float16',
                 'was_precipitating': 'float64', 'RWIS_site': 'str', 'month': 'float64',
                 'soilT_0_01m': 'float64', 'soilT_0_1m': 'float64'}

RF_OUTPUT_PATH = '../data/RF_predictions.csv'
LR_OUTPUT_PATH = '../data/LR_predictions.csv'
XGB_OUTPUT_PATH = '../data/GBT_predictions.csv'
