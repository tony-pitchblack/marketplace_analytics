import yaml
import numpy as np

def read_params(file_yaml):
 
    with open(file_yaml, "r") as file:
        data = yaml.safe_load(file)
    # y = exp(a1*x+b1)
    func1 = data['function']['func1']['name']
    a1 = data['function']['func1']['a1']
    b1 = data['function']['func1']['b1']
    par1 = [func1,a1,b1]
    # y = a2*(x - b2)*c2
    
    func2 = data['function']['func2']['name']
    a2 = data['function']['func2']['a2']
    b2 = data['function']['func2']['b2']
    c2 = data['function']['func2']['c2']
    par2 = [func2, a2, b2, c2] # , d4, e4

    n_point = data['price']['num_points']
    n_lev = data['price']['num_levels']
    trend = data['trend']
    trend = np.random.rand()*(trend[1]-trend[0])+trend[0]
    season_amp1 = data['seasonality']['A1']
    season_amp1= np.random.rand()*(season_amp1[1]-season_amp1[0])+season_amp1[0]
    season_amp2 = data['seasonality']['A2']
    season_amp2= np.random.rand()*(season_amp2[1]-season_amp2[0])+season_amp2[0]
    season_fi = data['seasonality']['fi']
    season_fi= np.random.rand()*(season_fi[1]-season_fi[0])+season_fi[0]
    period = 365
    
    season_amp = [season_amp1,season_amp2]

    return par1, par2, n_point, n_lev, trend, season_amp, season_fi, period