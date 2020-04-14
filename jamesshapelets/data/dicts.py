"""
Some dicts containing useful bits of information
"""
from jamesshapelets.definitions import *


# Some removals
univariate = os.listdir(DATA_DIR + '/interim/univariate')
remove = [
    'GesturePebbleZ1',
    'GesturePebbleZ2',
    'GestureMidAirD1',
    'DodgerLoopWeekend',
    'DodgerLoopGame',
    'DodgerLoopDay',
    'GestureMidAirD2',
    'ShakeGestureWiimoteZ',
    'MelbournePedestrian',
    'PickupGestureWiimoteZ',
    'AllGestureWiimoteX',
    'PLAID',
    'AllGestureWiimoteY',
    'GestureMidAirD3',
    'AllGestureWiimoteZ'
]


for r in remove:
    if r in univariate:
        univariate.remove(r)

multivariate = os.listdir(DATA_DIR + '/interim/multivariate')


ds_names = {
    'univariate': univariate,
    'multivariate': multivariate
}

learning_ts_shapelets = [
    'Adiac',
    'Beef',
    'BeetleFly',
    'BirdChicken',
    'ChlorineConcentration',
    'Coffee',
    'DiatomSizeReduction',
    'DP_Little',
    'DP_Middle',
    'DP_Thumb',
    'ECGFiveDays',
    'FaceFour',
    'GunPoint',
    'ItalyPowerDemand',
    'Lightning7',
    'MedicalImages',
    'MoteStrain',
    'MP_Little',
    'MP_Middle',
    'Otoliths',
    'PP_Little',
    'PP_Middle',
    'PP_Thumb',
    'SonyAIBORobotSurface1',
    'Symbols',
    'SyntheticControl',
    'Trace',
    'TwoLeadECG'
]
