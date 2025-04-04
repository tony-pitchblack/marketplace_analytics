from scipy.stats import poisson
import numpy as np
#from numpy import matlib as mb
#import math
#import matplotlib.pyplot as plt

class BaseLine:
    def __init__(self, params = ['exp', [-0.03, -0.001], [0.2, 6]], num_point=20, num_level=10,
                 trend = 1, seasonality = (1,0.3,0.5), period = 365):
        # params - функция и диапазон на каждый из параметров. Значения на параметры выбираются в диапазоне из равномерного распределения
        # num_point - количество  ценовых уровней
        # x_min, x_max - ценовые границы (сейчас все параметры заточены под этот дипазон) # TODO: перейти к диапазону [1, 2]
        # func - параметры модели заказов 'exp': y = a*exp(b*x), 
        # num_level - количество ценовых уровней
        # self_price - себестоимость товара
        # trend - тренд продаж 
        # seasonality - сезонность, состоит из амплитуды и фазы синусоиды. TODO: ввести разнообразие в сезонность, включить не периодические функции
        # period - время на котором вводится тренд и сезонность. period = 365 означает, что значении тренда = 1, обьем продаж вырастет за год в 2 раза
        # а сезонность - первая гармоника имеет частоту один год
        
        self._num_point = int(num_point)
        self.num_level = int(num_level)
        self._x_min = 0
        self._x_max = 1
        self._func = params[0]
        self._parameters = []
        if params[0] == 'exp':
            p = params[1]
            self._parameters.append(np.random.rand()*(p[1]-p[0]) + p[0])
            p = params[2]
            self._parameters.append(np.random.rand()*(p[1]-p[0]) + p[0])
        if params[0] == 'linexp':
            p = params[1]
            self._parameters.append(np.random.rand() * (p[1] - p[0]) + p[0])
            p = params[2]
            self._parameters.append(np.random.rand() * (p[1] - p[0]) + p[0])
            p = params[3]
            self._parameters.append(np.random.rand() * (p[1] - p[0]) + p[0])

        self.trend = trend
        self.seasonality = seasonality[0]
        self.seasonality.append(seasonality[1])

        self.period = period
        self.x = None # начальные точки, в start_points() изменяются
        self.start_points()
        self.lam = self.func(self.x)
        self.gen_start_responce()
        

    def start_points(self):
        # генерация начальных ценовых уровней. 
        # TODO: ввести распределение для вероятности появления цен. В середине диапазона появление цены вероятнее цем на краях
        
        threshold = 0.6 # вероятность того, что следующий ценовой уровень будет равен предыдущему
        levels = np.random.rand(self.num_level)*(self._x_max - self._x_min) +  self._x_min
        for i in range(self._num_point-self.num_level):
            rand1 = np.random.rand()
            if rand1<threshold:
                levels = np.r_[levels,levels[-1]]
            else:
                b = np.random.randint(self.num_level)
                levels = np.r_[levels,levels[b]]
        self.x = levels

    def gen_start_responce(self):
        # генерация продаж на заданном дипразоне ценовых уровней
        lam = self.get_lambda()
        # генерация заказов
        self.order = poisson.rvs(lam)

        self.lam = lam
        

    def func(self, data):
        # price model (lambda = f(price) = exp(a*price + b))
        # у всех функций есть параметры a и b
        a, b = self._parameters[0], self._parameters[1]
        if self._func == 'exp':
            return np.exp(-a*data+b)

        if self._func == 'linexp':
            # p[0] > 0 -- multiplier, p[1] < 0 -- x offset, p[2] > 0 -- steepness, p[3] -- y offset
            # moda = (p[1] + 1/p[2])
            # Oy intersect = -p[0]*p[1]*e^(p[1]*p[2]) + p[3] should be > 0 => p[1] < 0
            c = self._parameters[2]
            return a * (data - b) * np.exp(-c * (data - b))

    
    def get_lambda(self,*arg):
        # lambda с учетом сезонности и тренда (т.е. для каждого момента времени)
        if len(arg) == 0:
            time = np.arange(self.x.size)
            # trend = a + bx = a(1 + b/a*x)
            # a = lambda(t0)
            # b/a - угол наклона, т.е. сила тренда
            # x = time/period

            self.k = 1+time/self.period*self.trend

            #  сезонность 1-я гармоника
            sin = self.seasonality[0]*np.sin(2*np.pi*time/self.period+self.seasonality[2])
            #  сезонность 2-я гармоника
            sin += self.seasonality[1]*np.sin(4*np.pi*time/self.period+self.seasonality[2])
            b = sin>=0
            self.k[b] = self.k[b]*(1+sin[b])
            self.k[~b] = self.k[~b]/(1-sin[~b])
            lam = self.lam*self.k
        else:
            price = arg[0]
            if type(price) == list:
                price = np.array(price)
            time = self.x.size
            k = 1+time/self.period*self.trend

            #  сезонность
            sin = self.seasonality[0]*np.sin(2*np.pi*time/self.period+self.seasonality[2])
            #  сезонность 2-я гармоника
            sin += self.seasonality[1]*np.sin(4*np.pi*time/self.period+self.seasonality[2])
            if sin>=0:
                self.k = self.k*(1+sin)
            else:
                self.k = self.k/(1-sin)
            lam = self.func(price)*k
       
        return lam
    
    # вспомогательные функции
    def plot_trend(self, *arg):
        # если хотим вывести тренд за пределы горизонта исходных данных, передаем входной параметр - количество дней горизонта
        if len(arg) == 0:
            return self.k
        else:
            time = np.arange(self.x.size + arg[0])
            self.k = 1+time/self.period*self.trend

            #  сезонность 1-я гармоника
            sin = self.seasonality[0]*np.sin(2*np.pi*time/self.period+self.seasonality[2])
            #  сезонность 2-я гармоника
            sin += self.seasonality[1]*np.sin(4*np.pi*time/self.period+self.seasonality[2])
            b = sin>=0
            self.k[b] = self.k[b]*(1+sin[b])
            self.k[~b] = self.k[~b]/(1-sin[~b])
            return self.k

        

    



