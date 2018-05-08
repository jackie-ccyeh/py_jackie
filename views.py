from __future__ import division
from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse
from datetime import datetime
from myapp.models import mdesign_data1
import pandas as pd
import numpy as np


def sayhello(request):
    return HttpResponse('Hello Django, Jackie!')

def hello2(request, username):
    return HttpResponse('hello ' + username)

def hello3(request, username):
    now = datetime.now()
    return render(request, 'hello3.html', locals())

def hello4(request, username, kk):
    now = datetime.now()
    return render(request, 'hello4.html', locals())

def listone(request):
    try:
        unit = mdesign_data1.objects.get(id='100')
    except:
        errormessage = 'reading error'
    return render(request, 'listone.html', locals())

def listall(request):
    mdesigns = mdesign_data1.objects.all().order_by('id')
    return render(request, 'listall.html', locals())


def MaxMinNor(x, Max, Min):
    """
    function of max min normalization
    """
    x = (x - Min) / (Max - Min)
    return x

def post(request):
    if request.method == 'POST':
        df_alldata = pd.DataFrame(list(mdesign_data1.objects.all().values()))
        # print(str(df_alldata.shape) + 'haha') for debug
        df_alldata.loc[:, 'rated_speed'] = np.float32(df_alldata.loc[:, 'rated_speed'])
        #df_alldata = df_alldata[df_alldata.loc[:, 'rated_speed'] > 1000]
        #df_alldata = df_alldata[df_alldata.loc[:, 'rated_speed'] < 10000]

        df_alldata.loc[:, 'voltage'] = np.float32(df_alldata.loc[:, 'voltage'])
        df_alldata.loc[:, 'output_power'] = np.float32(df_alldata.loc[:, 'output_power'])
        df_alldata.loc[:, 'equivalent_model_depth'] = np.float32(df_alldata.loc[:, 'equivalent_model_depth'])

        df2 = df_alldata.copy(deep=True)
        speed_max = np.float32(np.max(df2.loc[:, 'rated_speed']))
        speed_min = np.float32(np.min(df2.loc[:, 'rated_speed']))
        voltage_max = np.float32(np.max(df2.loc[:, 'voltage']))
        voltage_min = np.float32(np.min(df2.loc[:, 'voltage']))
        power_max = np.max(df2.loc[:, 'output_power'])
        power_min = np.min(df2.loc[:, 'output_power'])
        size_max = np.max(df2.loc[:, 'equivalent_model_depth'])
        size_min = np.min(df2.loc[:, 'equivalent_model_depth'])
        print(str(type(df2.loc[3, 'voltage'])) + str(type(df2.loc[3, 'output_power'])) +
              str(type(df2.loc[3, 'rated_speed'])) + str(type(df2.loc[3, 'equivalent_model_depth'])))
        df2.loc[:, 'voltage'] = MaxMinNor(df2.loc[:, 'voltage'],
                                              voltage_max, voltage_min)
        df2.loc[:, 'output_power'] = MaxMinNor(df2.loc[:, 'output_power'],
                                              power_max, power_min)
        df2.loc[:, 'rated_speed'] = MaxMinNor(df2.loc[:, 'rated_speed'],
                                              speed_max, speed_min)
        df2.loc[:, 'equivalent_model_depth'] = MaxMinNor(df2.loc[:, 'equivalent_model_depth'],
                                              size_max, size_min)

        v = np.float32(request.POST['voltage'])
        p = np.float32(request.POST['power'])
        d = np.float32(request.POST['size'])
        s = np.float32(request.POST['speed'])
        v1 = (v - voltage_min)/(voltage_max-voltage_min)
        p1 = (p - power_min) / (power_max - power_min)
        d1 = (d - size_min) / (size_max - size_min)
        s1 = (s - speed_min) / (speed_max - speed_min)
        new_point = np.array([[v1, p1, d1, s1]], dtype = np.float32)
        data_points = df2.loc[:, ['voltage', 'output_power', 'equivalent_model_depth', 'rated_speed']].values

        data_points = data_points[data_points != np.array(None)]
        data_points = data_points.reshape(-1, 4)
        print(str(data_points.shape) + 'haha')
        k = np.sum(np.square(new_point - data_points), axis=1)
        k1 = dict(enumerate(k))
        k2 = sorted(k1.items(), key=lambda item: item[1])
        ind = []
        for i in range(len(k2[:5])):
            ind.append(k2[:5][i][0])
        aa = df_alldata.loc[ind, ['voltage', 'average_input_current', 'input_power', 'output_power', 'equivalent_model_depth', 'number_of_conductors_per_slot', 'rated_speed', 'steel', 'magnet']].values
    return render(request, 'post.html', locals())

'''
    old method, but not correct: 
        data_1 = np.square(v - df_alldata.loc[:, ['voltage']]).join(np.square(d - df_alldata.loc[:, ['equivalent_model_depth']]))\
                .join(np.square(p - df_alldata.loc[:, ['output_power']])).join(np.square(s - df_alldata.loc[:, ['rated_speed']]))
        ind = data_1.sort_values('rated_speed').iloc[range(0, 100), :].sort_values('output_power').iloc[range(0, 50),
              :].sort_values('equivalent_model_depth').iloc[range(0, 25), :].sort_values('voltage').iloc[range(0, 5),
              :].index
'''
