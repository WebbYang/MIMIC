import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import peakutils
from numpy.polynomial.polynomial import polyfit
import plotly.express as px
from scipy import signal
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

# @dataclass
# class Person:
#     name: str
#     age: int


def assign_t(data):
    '''
    transfer time from cylic format
    '''
    data = data.values
    try:
        diff = data[1:]-data[:-1]
    except:
        
        data = data.astype(int)
        diff = data[1:]-data[:-1]
    shift = 0
    #outlier = []
    for i,item in enumerate(diff):
        if item>8:
            print(f'No {i+1}: {data[i+1]}, {data[i+2]}, {item}')
            data[i+1] = data[i]+4
            
    for i,item in enumerate(diff):
        if item<-65000:
            shift += 65500
#         elif item<-1000: # break if new start
#             data = data[:i]
#             break
        elif item<-90: #<0
            shift += 100
        
#         elif item>8:
#             print(f'No {i+1}: {data[i+1]}, {data[i+2]}, {item}')
#             data[i+1] = data[i]+4
#             #shift -= 100
        data[i+1]+=shift
    

def signal_filter(data, col, fc_l, fc_h, fs):
    sos = signal.butter(4, fc_h, 'lp', fs=fs, output='sos') # to compare 20211112, cutoff 
    data[col+'_filt'] = signal.sosfiltfilt(sos, data[col]) #10
    sos = signal.butter(4, fc_l, 'lp', fs=fs, output='sos')
    data[col+'_baseline'] = signal.sosfiltfilt(sos, data[col+'_filt'])
    data[col+'_filt'] = data[col+'_filt'] - data[col+'_baseline']
    
    for i in range(4):
        sos = signal.butter(i, fc_h, 'lp', fs=fs, output='sos')
        data[col+f'_filt_{i}'] = signal.sosfiltfilt(sos, data[col])
        data[col+f'_filt_{i}'] = data[col+f'_filt_{i}'] - data[col+'_baseline']

    
# https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data 

def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def find_continuous_init(out_lier):
    
    old_item = out_lier[0]
    buffer = []
    
    for item in out_lier[1:]:
        buffer.append(old_item)
        if item > 30 and item - old_item < 5:
            print(buffer)
            if len(buffer)>2:
                return buffer[0]
        else:
            buffer.pop()
        old_item = item
    return item

def dynamic_table(bp_idx, ppg_idx, min_thrd=0.01, max_thrd=0.2):
    i, j = 0, 0
    bp_i, ppg_i = [], []
    while i<len(bp_idx) and j<len(ppg_idx):
        if ppg_idx[j]-bp_idx[i]<min_thrd: # -0.3
            j+=1
        else:
            if ppg_idx[j]-bp_idx[i] < max_thrd: #0.2
                bp_i.append(i)
                ppg_i.append(j)
                i+=1
                j+=1
            else:
                i+=1
                
    return bp_i, ppg_i

def get_bp_idx(filename, data, save=False):
    bp_idx = data['bp_base_idx']
    #bp_idx = np.array(bp_idx)/250
    try:
        bp_idx = np.array(data['t'])[bp_idx]/1000
    except:
        bp_idx = np.array(data['bp_t'])[bp_idx]/1000
    bp_itvl = bp_idx[1:]-bp_idx[:-1]
    
    unmask = mad_based_outlier(bp_itvl)
    mask = ~ unmask
    outlier = np.arange(len(mask))[unmask]
    mask = [False]+[item for item in mask]

    for i in outlier:
        mask[i] = False

    bp_idx2 = bp_idx[mask]
    #data['bp_mask'] = mask
    
    plt.figure()
    plt.plot(bp_idx[1:], bp_itvl, 'o',  markerfacecolor='none')
    plt.scatter(bp_idx[1:][outlier], bp_itvl[outlier], color='r')
    plt.title('BP Interval Scatter')
    plt.xlabel('Time of End point (s)')
    plt.ylabel('Pulse Interval(s)')
    plt.grid()
    name = filename.split('/')[-1].split('.')[0]
    if save:
        plt.savefig(f'PTT_{name}.png')
    plt.show()
    
    return bp_idx2
    
def get_ppg_idx(filename, data, save=False):
    ppg_idx = data['ppg_base_idx']
    #ppg_idx = np.array(ppg_idx)/250
    try:
        ppg_idx = np.array(data['t'])[ppg_idx]/1000
    except:
        ppg_idx = np.array(data['ppg_t'])[ppg_idx]/1000
    ppg_itvl = ppg_idx[1:]-ppg_idx[:-1]
            
    unmask = mad_based_outlier(ppg_itvl)
    mask = ~ unmask
    outlier = np.arange(len(mask))[unmask]
    mask = [False]+[item for item in mask]
    ppg_idx2 = ppg_idx[mask]
    #data['ppg_mask'] = mask
    
    plt.figure()
    plt.plot(ppg_idx[1:], ppg_itvl, 'o',  markerfacecolor='none')
    plt.scatter(ppg_idx[1:][outlier], ppg_itvl[outlier], color='r')
    plt.title('PPG Interval Scatter')
    plt.xlabel('Time of End point (s)')
    plt.ylabel('Pulse Interval(s)')
    plt.grid()
    name = filename.split('/')[-1].split('.')[0]
    if save:
        plt.savefig(f'PTT_{name}.png')
    plt.show()
    
    return ppg_idx2#, new_ppg_idx

def parse_DrHealth_data(fname, outname=None):
    START = 0 #1000 #4000 #4500
    print(f'Parse {fname}')
    scaler = StandardScaler()
    data = {}
    df = pd.read_csv(fname, header=None, skiprows=10) #skiprows=1
    aligned = False
    bp_estimate = False
    if len(df.columns)==3:
        df.columns = ['t','bp_v','ppg_v']
        aligned = True
        df['ppg_v'] = 65535 - df['ppg_v']
    elif len(df.columns)==4:
        df.columns = ['bp_t','bp_v','ppg_t','ppg_v']
        print('Now bp ...')
        assign_t(df['bp_t'])
        print('Now ppg ...')
        assign_t(df['ppg_t'])
        df['ppg_v'] = 65535 - df['ppg_v']
        df['bp_v'] = (df['bp_v']-(-1580))*200/(24570-(-1580))
        bp_estimate = True
    elif len(df.columns)==6:
        df.columns = ['ecg_t','ecg_v','ppg_t','ppg_v','bp_t','bp_v']
        print('Now ecg ...')
        assign_t(df['ecg_t'])
        print('Now ppg ...')
        assign_t(df['ppg_t'])
        print('Now bp ...')
        assign_t(df['bp_t'])
        
    #df['ppg_v'] = 65535 - df['ppg_v']
    #df['bp_v'] = (df['bp_v']-(-414))*300/(9312-(-414))
    
    # scan bp_v rising -> shift start index
    bp_arr = df['bp_v'].values
    #
#     if 'Ray' in fname:
#         END = -2500
#     else:
#         END = -1
    END = -1
    
    if bp_estimate:       
        bp_maxidx = np.argmax(bp_arr[:5000])
        START += bp_maxidx
#     else:
#         diff = bp_arr[1:] - bp_arr[:-1]
    
#         for i,item in enumerate(diff):
#             if item<0:
#                 break
#         if i>1000:
#             #START += START
#             START += i        
    for col in df.columns:
        #data[col] = df[col].iloc[START:-START].values
        data[col] = df[col].iloc[START:END].values
    #data['ppg_v'] = scaler.fit_transform(data['ppg_v'].reshape(-1,1)).flatten()
    #data['ppg_v'] *= 0.9*(np.percentile(data['bp_v'],75)-np.percentile(data['bp_v'],25))   
    
    if df['bp_t'].iloc[1]-df['bp_t'].iloc[0]==8:
        data['fs'] = 125
    elif df['bp_t'].iloc[1]-df['bp_t'].iloc[0]==2:
        data['fs'] = 500
    else:
        data['fs'] = 250

    fig, ax = plt.subplots(figsize=(30,5))
    for col in df.columns[::2]:
        col = col[:-2]
        signal_filter(data, f'{col}_v', 0.2, 20, data['fs']) # reimplement to input args 20211029
        # scaling for oscillogram
        if col=='bp' or col=='ecg':
            data[f'{col}_v_filt'] = scaler.fit_transform(data[f'{col}_v_filt'].reshape(-1,1)).flatten()
        if col=='ppg':
            data['ppg_v_filt'] = scaler.fit_transform(data['ppg_v_filt'].reshape(-1,1)).flatten()
            #data['ppg_v_filt'] *= 1.5*(np.percentile(data['bp_v_filt'],75)-np.percentile(data['bp_v_filt'],25))
        if aligned:
            plt.plot(data['t']/1000, data[f'{col}_v_filt'], label=col)
        else:
            plt.plot(data[f'{col}_t']/1000, data[f'{col}_v_filt'], label=col)
    plt.rcParams['axes.unicode_minus']=False
    plt.title(fname.split('/')[-1].split('.')[0])
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid()
    if outname is not None:
        plt.savefig(f'processed_signal_{outname}.png')
    #plt.close()

    return data

def plot_aligned(bp_idx, ppg_idx, bp_i, ppg_i, name):
    x = ppg_idx[ppg_i]
    y = ppg_idx[ppg_i]-bp_idx[bp_i]
    #y1 = new_ppg_idx[ppg_i]-bp_idx[bp_i]
    fig, ax = plt.subplots(figsize=(30,5))
    plt.plot(x, y, '-o') #, label='original'
    #plt.plot(x, y1, '-x', label='fixed')
    plt.xlabel('Pulse Valley Time (sec)')
    plt.ylabel('Pulse Transit Time (sec)')
    #plt.legend()
    plt.title(f'{name}')
    plt.grid()
    plt.savefig(f'PTT_{name}.png')
    plt.show()
    return y, x


def re_seg_hd(ppg_data, thrd=0.1, amp_ratio=0.5, min_dist=10, plot=True, shift=0, save_fig=False, out=''):
    '''
    thrd: peakutils parameter for peak detection threshold
    amp_ratio: self-designed condition for newly-qualified pulse relative to the previuos one 
    '''
    if shift!=0:
        ppg_data = ppg_data[shift:]
#     if abs(ppg_data[1]-ppg_data[0])<5:
#         min_dist = 5 #20
#         thrd=0.01
#         amp_ratio=0.4
    v_idx = peakutils.indexes(-ppg_data, thres=thrd, min_dist=min_dist)
    true_v_idx = []
    #base_level = np.percentile(ppg_data, 50)
    for idx in v_idx:
        try:
            if keep_positive(ppg_data[idx:idx+min_dist],-1):#40 ppg_data[idx]<base_level+0.2 and 
                true_v_idx.append(idx)
            else:
                #print('I lost one: ',idx)
                continue
        except:
            if len(ppg_data)-idx>0:
                if ppg_data[-1]>ppg_data[idx]: #ppg_data[idx]<base_level and 
                    true_v_idx.append(idx)
            
    true_p_idx = []
    amp = []
    i=0
    while i<len(true_v_idx)-1:
        amp_thrd = 0.1 if len(amp)<3 else 0.2*amp_ratio*amp[-2]+0.8*amp_ratio*amp[-1] #0.4
        seg = list(ppg_data[true_v_idx[i]:true_v_idx[i+1]])
        p_idx = [seg.index(max(seg))]
        found = False
        for p in p_idx:
            if p>min_dist:
                # peak should be higher than the left and right side of vallies by at least amp threshold
                amp_fall = ppg_data[true_v_idx[i]+p]-ppg_data[true_v_idx[i+1]]
                amp_rise = ppg_data[true_v_idx[i]+p]-ppg_data[true_v_idx[i]]
                pred_peak = true_v_idx[i]+p
                if amp_fall>amp_thrd and amp_rise>amp_thrd and keep_positive(-ppg_data[pred_peak:pred_peak+min_dist],-1):
                    true_p_idx.append(true_v_idx[i]+p)
                    #amp.append(amp_rise) #20211015
                    amp.append(amp_fall)
                    found = True
                    break
            
        if not found:
            # if peak not found, then pop the next valley or current valley depends on who's lower
            #if ppg_data[pred_peak-20]-ppg_data[pred_peak]<ppg_data[pred_peak+20]-ppg_data[pred_peak]:
            if p<=min_dist:
                #print('Found not good! ',true_v_idx[i])
                if ppg_data[true_v_idx[i]]<ppg_data[true_v_idx[i+1]]:
                    true_v_idx.pop(i+1)
                else:
                    true_v_idx.pop(i)
            elif amp_fall<amp_rise:# or keep_positive(-ppg_data[pred_peak:pred_peak+20])
                #print('Pop next: ', true_v_idx[i+1])
                true_v_idx.pop(i+1)                
            else:
                #print('Pop current: ', true_v_idx[i])
                true_v_idx.pop(i)               
            i-=1            
                
        i+=1                  
        
    if plot:    
        plt.figure(figsize=(20,3))
        plt.plot(ppg_data)
        plt.scatter(true_v_idx, ppg_data[true_v_idx],facecolors='none', edgecolors='r')
        plt.scatter(true_p_idx, ppg_data[true_p_idx],facecolors='none', edgecolors='g')
        #i_test = 1016#[12909, 12948]
        #plt.scatter(i_test, ppg_data[i_test],color='b')
        if save_fig:
            plt.savefig(f'peak_calling_{out}.png')
        plt.show()
    
    amp = []
    itvl = []
    for i in range(len(true_p_idx)):
        amp.append(ppg_data[true_p_idx[i]] - ppg_data[true_v_idx[i]])
        itvl.append(true_v_idx[i+1] - true_v_idx[i])
    
    return amp, itvl, np.array(true_v_idx[:-1])+shift, np.array(true_p_idx)+shift

def re_seg_hd_v3(data, key, thrd=0.1, amp_ratio=0.5, min_dist=10, plot=True, shift=0, save_fig=False, out=''):
    '''
    thrd: peakutils parameter for peak detection threshold
    amp_ratio: self-designed condition for newly-qualified pulse relative to the previuos one 
    '''
    if key=='ppg':
        ppg_data = data['ppg_v_filt']
        time_data = data['ppg_t']
        scaler = StandardScaler()
        ppg_data = scaler.fit_transform(ppg_data.reshape(-1,1)).flatten()
        
    elif key=='bp':
        ppg_data = data['bp_v_filt']
        time_data = data['bp_t']
        scaler = StandardScaler()
        ppg_data = scaler.fit_transform(ppg_data.reshape(-1,1)).flatten()
        
    if shift!=0:
        ppg_data = ppg_data[shift:]

    v_idx = peakutils.indexes(-ppg_data, thres=thrd, min_dist=min_dist)
    true_v_idx = []
    
    for idx in v_idx:
        try:
            if keep_positive(ppg_data[idx:idx+min_dist],-1):#40 ppg_data[idx]<base_level+0.2 and 
                true_v_idx.append(idx)
            else:
                #print('I lost one: ',idx)
                continue
        except:
            if len(ppg_data)-idx>0:
                if ppg_data[-1]>ppg_data[idx]: #ppg_data[idx]<base_level and 
                    true_v_idx.append(idx)
            
    true_p_idx = []
    amp = []
    i=0
    while i<len(true_v_idx)-1:
        amp_thrd = 0.5 if len(amp)<3 else amp_ratio*amp[-1] #0.4
        seg = list(ppg_data[true_v_idx[i]:true_v_idx[i+1]])
        p_idx = [seg.index(max(seg))]
        found = False
        for p in p_idx:
            if p>min_dist:
                # peak should be higher than the left and right side of vallies by at least amp threshold
                amp_fall = ppg_data[true_v_idx[i]+p]-ppg_data[true_v_idx[i+1]]
                amp_rise = ppg_data[true_v_idx[i]+p]-ppg_data[true_v_idx[i]]
                pred_peak = true_v_idx[i]+p
                if amp_fall>amp_thrd and amp_rise>amp_thrd and keep_positive(-ppg_data[pred_peak:pred_peak+min_dist],-1):
                    true_p_idx.append(true_v_idx[i]+p)
                    #amp.append(amp_rise) #20211015
                    amp.append(amp_fall)
                    found = True
                    break
            
        if not found:
            # if peak not found, then pop the next valley or current valley depends on who's lower
            #if ppg_data[pred_peak-20]-ppg_data[pred_peak]<ppg_data[pred_peak+20]-ppg_data[pred_peak]:
            if p<=min_dist:
                #print('Found not good! ',true_v_idx[i])
                if ppg_data[true_v_idx[i]]<ppg_data[true_v_idx[i+1]]:
                    true_v_idx.pop(i+1)
                else:
                    true_v_idx.pop(i)
            elif amp_fall<amp_rise:# or keep_positive(-ppg_data[pred_peak:pred_peak+20])
                #print('Pop next: ', true_v_idx[i+1])
                true_v_idx.pop(i+1)
            else:
                #print('Pop current: ', true_v_idx[i])
                true_v_idx.pop(i)
            i-=1            
                
        i+=1                  
        
    if plot:
        # todo: cond = (time_data >1000) & (time_data <5000)
        plt.figure(figsize=(20,3))
        plt.title(key)
        plt.plot(time_data/1000, ppg_data)
        plt.scatter(time_data[true_v_idx]/1000, ppg_data[true_v_idx],facecolors='none', edgecolors='r')
        plt.scatter(time_data[true_p_idx]/1000, ppg_data[true_p_idx],facecolors='none', edgecolors='g')
        plt.xlabel('Time (sec)')
        #i_test = 1016#[12909, 12948]
        #plt.scatter(i_test, ppg_data[i_test],color='b')
        if save_fig:
            plt.savefig(f'peak_calling_{out}.png')
        plt.show()
    
    amp = []
    itvl = []
    for i in range(len(true_p_idx)):
        amp.append(ppg_data[true_p_idx[i]] - ppg_data[true_v_idx[i]])
        itvl.append(true_v_idx[i+1] - true_v_idx[i])
    
    return amp, itvl, np.array(true_v_idx[:-1])+shift, np.array(true_p_idx)+shift

def slope(data, fs):
    points=int(12*fs/250)
    padding = int(points/2)
    #return np.array([0]*padding+list(data[points:]-data[:-points])+[0]*(points-padding))
    newdata = np.array([0]*padding+list(data)+[0]*padding)
    return newdata[points:]-newdata[:-points]

def detect_fp(peaks, fs):
    diff = peaks[1:] - peaks[:-1]
    #input(diff)
    thrd = 100*fs/250
    for item in diff:
        if item<thrd: # 250Hz index, so x4 -> 400ms
            return True
    return False

def detect_fn(peaks, fs):
    diff = peaks[1:] - peaks[:-1]
    #input(diff)
    thrd = 500*fs/250
    for item in diff:
        if item>thrd: # 2000ms
            return True
    if len(peaks)<5:
        return True
    return False

def slope_peak(diff_data, fs=250, thres=0.3, min_dist=50): #50
    strictFlag = False
    res = peakutils.indexes(diff_data, thres, min_dist)
    
    #====== for ECG ========
    to_pop = []
    for i,item in enumerate(res[:-1]):
        if res[i+1] - item < 50:
            to_pop.append(i+1)
    res = np.array([item for i,item in enumerate(res) if i not in to_pop])
    #====== for ECG ========
        
    while detect_fp(res, fs):
        thres+=0.1
        res = peakutils.indexes(diff_data, thres, min_dist)
        strictFlag = True
        
    if strictFlag==False:
        while detect_fn(res, fs):
            thres-=0.1
            res = peakutils.indexes(diff_data, thres, min_dist)
    else:
        while detect_fn(res, fs):
            thres-=0.04
            res = peakutils.indexes(diff_data, thres, min_dist)
            if thres<=0:
                break
        
    return res

def edge_detect(data, feat='bp'):
    sig = data[f'{feat}_v_filt']
    fs = data['fs']
    
    diff_sig = slope(sig, fs) #, points=kw['pts'][feat]
    data[f'{feat}_diff_sig'] = diff_sig
    #====== for ECG ========
    if feat=='bp':
        return slope_peak(diff_sig, fs=fs, thres=0.8, min_dist=20) #temp 20220407
    #====== for ECG ========
    return slope_peak(diff_sig, fs=fs)

def ppt_pipe(data, min_thrd=0.01, max_thrd=0.2, method='edge'):
    if method=='edge':
        bp_edge_idx = edge_detect(data, 'bp')  
        ppg_edge_idx = edge_detect(data, 'ppg')
        data['bp_edge_idx'] = bp_edge_idx
        data['ppg_edge_idx'] = ppg_edge_idx
        bp_edge_idxt, ppg_edge_idxt = data['bp_t'][bp_edge_idx]/1000, data['ppg_t'][ppg_edge_idx]/1000        
        edge_bp_i, edge_ppg_i = dynamic_table(bp_edge_idxt, ppg_edge_idxt,min_thrd, max_thrd)
        data['ppg_edge_idxt'] = data['ppg_t'][ppg_edge_idx[edge_ppg_i]]
        data['bp_edge_idxt'] = data['bp_t'][bp_edge_idx[edge_bp_i]]
        ptt =  data['ppg_edge_idxt'] - data['bp_edge_idxt']
        data['edge_ptt'] = ptt
        if 'ecg_t' in data.keys():
            data['ecg_edge_idx'] = ecg_edge_idx = edge_detect(data, 'ecg')
            ecg_edge_idxt = data['ecg_t'][ecg_edge_idx]/1000
            #min_thrd, max_thrd = 0.15, 0.8
            min_thrd, max_thrd = 0.25, 0.8
            edge_ecg_i, edge_ppg_i = dynamic_table(ecg_edge_idxt, ppg_edge_idxt,min_thrd, max_thrd)
            data['ecg_edge_idxt'] = data['ecg_t'][ecg_edge_idx[edge_ecg_i]]
            data['ppg_edge_idxt_e'] = data['ppg_t'][ppg_edge_idx[edge_ppg_i]]
            ptt_e = data['ppg_edge_idxt_e'] - data['ecg_edge_idxt']
            data['edge_ptt_ecg'] = ptt_e
            
    else:
        bp_base_idx = data[f'bp_{method}_idx']
        ppg_base_idx = data[f'ppg_{method}_idx']       
        bp_base_idxt, ppg_base_idxt = data['bp_t'][bp_base_idx]/1000, data['ppg_t'][ppg_base_idx]/1000        
        base_bp_i, base_ppg_i = dynamic_table(bp_base_idxt, ppg_base_idxt, min_thrd, max_thrd)
        data[f'ppg_{method}_idxt'] = data['ppg_t'][ppg_base_idx[base_ppg_i]]
        data[f'bp_{method}_idxt'] = data['bp_t'][bp_base_idx[base_bp_i]]
        ptt =  data[f'ppg_{method}_idxt'] - data[f'bp_{method}_idxt']
        data[f'{method}_ptt'] = ptt
        
    #cond = ptt<200
    #plt.plot(data['ppg_t'][ppg_edge_idx[edge_ppg_i]][cond]/1000, ptt[cond], '-o')
    plt.figure(figsize=(15,3))
    plt.plot(data[f'ppg_{method}_idxt']/1000, ptt, '-o', label='bp-ppg')
    if 'ecg_t' in data.keys():
        plt.plot(data[f'ecg_{method}_idxt']/1000, ptt_e, '-o', label='ecg-ppg')
    plt.title(f'{method} Detection PTT')
    plt.xlabel(f'PPG {method} time (sec)')
    plt.ylabel('PTT (ms)')
    plt.legend()
    plt.show()
    return ptt


def keep_positive(seg, thrd=0):
    for i,val in enumerate(seg[1:]):
        if val-seg[i]<thrd:
            return False
    return True


def trigger_idx(data):
    data = data.copy()
    diff = np.abs(data[1:]-data[:-1])
    #thrd = np.max(diff)/10
    thrd = np.sort(diff)[-4]/10
    print(f'Thred: {thrd}')
    idxes = []
    old_i = 0
    for i,item in enumerate(diff):
        if item>thrd and i-old_i>10:         
            old_i = i
            idxes.append(i)

    return idxes

def smooth_sig(data):
    '''
    Deal with Ray's motor signal 20210729
    '''
    #data = data.copy() # if for direct use
    #idxes = peakutils.indexes(data, thres=0.2, min_dist=1000) #0.01
    idxes = trigger_idx(data)
    diff = np.abs(data[1:]-data[:-1])
    diff_m = np.mean(diff)
    print(f'mean diff: {diff_m}')
    for idx in idxes:
        # deal with not only 1 trigger sig
        i=[]
        #print(idx/488, diff[idx-2:idx+3])
        for k in range(-2,5):
            try:
                if diff[idx+k]>diff_m:
                    i.append(k)
            except:
                print(f'out of bound: {k}')
        for k in i:
            data[idx+k] = (data[idx+k-1]+data[idx+i[-1]+1])/2

    return idxes#, data

def interpolate(k,s,e,val_s,val_e):
    return val_s*(e-k)/(e-s)+val_e*(k-s)/(e-s)
    
def smooth_sig_BP(data):
    diff = np.array(data[1:]-data[:-1])
    RISE_FLAG = False
    Fall_FLAG = False

    fake_itvls = []
    for i,item in enumerate(diff):
        if item>10 and not RISE_FLAG:
            fake_start = i-1
            RISE_FLAG = True
        if item<-10:
            if RISE_FLAG:
                Fall_FLAG = True
                RISE_FLAG = False
        if Fall_FLAG and abs(item)<3:
            fake_end = i+1
            fake_itvls.append((fake_start, fake_end))          
            Fall_FLAG = False
                
    for item in fake_itvls:
        s,e = item[0], item[1]
        #print(s,e)
        val_s,val_e = data[s], data[e]
        #print(data[s:e])
        for k in range(s+1,e):            
            data[k] = interpolate(k,s,e,val_s,val_e)
        #print(data[s:e])
        #print('---------')
                
    return np.array([item[0] for item in fake_itvls])


def parse_data(file_path, filt=False):
    data = {}
    df = pd.read_csv(file_path, skiprows=39)
    # deal with unmatched column names
    if df.columns[1]=='AI 0':
        col_name = df.columns.tolist()
        col_name[1] = 'AI0'
        col_name[2] = 'AI1'
        df.columns = col_name
    
    start_idx = df['AI0'][df['AI0']>0].index[0]
    data['raw_AI0'] = df['AI0'].values.copy()[start_idx:]
    data['signal_A0'] = df['AI0'].values.copy()[start_idx:]
    data['signal_A1'] = df['AI1'].values.copy()[start_idx:]
    len_sig = len(data['signal_A1'])

#     def smooth_sig(data):
#         '''
#         Deal with Ray's motor signal 20210729
#         '''
# #         idxes = peakutils.indexes(data, thres=0.6) #0.01
# #         diff = np.abs(data[1:]-data[:-1])
# #         diff_m = np.mean(diff)
# #         for idx in idxes[:-1]:
# #             # deal with not only 1 trigger sig
# #             i=1
# #             while diff[idx+i-1]<diff_m:
# #                 i+=1
# #             data[idx] = (data[idx-1]+data[idx+i])/2
# #             if i>1:
# #                 for j in range(1,i):
# #                     data[idx+j] = (data[idx+j-1]+data[idx+i])/2
#         idxes = peakutils.indexes(data, thres=0.6) #0.01
#         diff = np.abs(data[1:]-data[:-1])
#         diff_m = np.mean(diff)
#         print(f'mean diff: {diff_m}')
#         for idx in idxes:
#             # deal with not only 1 trigger sig
#             i=[]
#             print(idx/250, diff[idx-2:idx+3])
#             for k in range(-2,3):
#                 if diff[idx+k]>diff_m:
#                     i.append(k)
#             for k in i:
#                 data[idx+k] = (data[idx+k-1]+data[idx+i[-1]+1])/2
#         return idxes
    
    if not 'ppg' in file_path.split('/')[-1]:
        data['trigger_sig'] = smooth_sig_BP(data['signal_A0']) #, burst_idx
        smooth_sig_BP(data['signal_A1']) #, burst_idx
        #sos = signal.butter(10, 10, 'lp', fs=122, output='sos') #488
    else:
        data['trigger_sig'] = smooth_sig(data['signal_A0'])
        sos = signal.butter(10, 15, 'lp', fs=250, output='sos') #
        # bandpass setting
        #sos = signal.butter(2, [0.2,15], 'bandpass', fs=250, output='sos')
        data['signal_A1'] = signal.sosfilt(sos, data['signal_A0'])
        sos = signal.butter(4, 0.2, 'lp', fs=250, output='sos')
        data['sig_baseline'] = signal.sosfiltfilt(sos, data['signal_A1'])
        data['signal_A1'] = data['signal_A1'] - data['sig_baseline']
        scaler = StandardScaler()
        data['signal_A1'] = scaler.fit_transform(data['signal_A1'].reshape(-1,1)).flatten()
        
        # decompose setting
#     if filt:
#         data['signal_A1'] = signal.sosfiltfilt(sos, data['signal_A0'])
#         sos = signal.butter(4, 0.2, 'lp', fs=250, output='sos')
#         data['sig_baseline'] = signal.sosfiltfilt(sos, data['signal_A1'])
#         data['signal_A1'] = data['signal_A1'] - data['sig_baseline']
        
        
#         start_pt = data['signal_A1'][0]
#         left_extend = np.append([start_pt]*int(250/0.2), data['signal_A1'])
#         baseline = signal.sosfilt(sos, left_extend)
#         data['sig_baseline'] = baseline[int(250/0.2):] #np.append(baseline[5*125:],[baseline[-1]]*5*125)
#         data['signal_A1'] = data['signal_A1'] - data['sig_baseline']
            
#     p_max_t = df[data['signal_A0']==data['signal_A0'].max()]['Time'].iloc[0]
#     cond_roi = (df['AI0']>=25)&(df['Time']<=p_max_t)&(df['Time']>4)
#     data['signal_A0'] = data['signal_A0'][cond_roi]
#     data['signal_A1'] = data['signal_A1'][cond_roi]
    
    #data['raw_A0'] = df['AI0'][cond_roi].values
    #data['raw_A1'] = df['AI1'][cond_roi].values

    fp = open(file_path,'r')    
    for i in range(37):
        line = fp.readline()
        if i==30 or i==31:
            # parse ref SYS, DIA
            k,v = line.split(',')[:2]
            data[k] = v     
        if i>=34:
            # parse device SYS, DIA
            k,v = line.split(',')[:2]
            data[k] = v
    fp.close()
    
    return data

'''
global variable:
    data_XX: signals and pressure
    osw_p: external pressure, log(amp)
constant:
    amp_thrd: 10
'''
amp_thrd = 10
class process_peak():
    def __init__(self, files, xls=None, file_map=None):
        self.data_HL = {}
        self.osw_p = {}
        self.personal_param = {}

        if xls is not None:
            df = pd.read_excel(xls, sheet_name='工作表2',usecols=range(8))
            df.dropna(inplace=True)
        
            FILE_ROOT = '/'.join(files[0].split('/')[:-1])

            file_key = [f'{FILE_ROOT}/BP_{m}_{int(n):02}.csv' for m,n in zip(df['Name'], df['Trial'])]
            file_val = [f'{a}_normal' if b==0 else f'{a}_slow' for a, b in zip(df['DIA (Clinical)'].tolist(), df['SYS (Clinical)'].tolist())]
            
            self.file_map = {k:v for k,v in zip(file_key, file_val)}
            
        else:
            if file_map is not None:
                self.file_map = file_map
            else:
                print('Please provide input: <xls> or <file_map>')
                return -1
        

        self.grp_name_list = []
        for file in files:
            name = self.file_map[file]
            if name not in self.grp_name_list:
                self.grp_name_list.append(name)
        
    def main_process(self):
        for grp_name in self.grp_name_list:
            files_grp = [k for k,v in self.file_map.items() if grp_name==v]
            if len(files_grp)>0:
                print(grp_name)
                self.pulse_segment(files_grp, plot=False)
                if not is_contain(['arm','slow','normal','ppg'], grp_name): # temperarily block the further analysis
                    self.oscillo_extract(files_grp)
                    to_see = {k:self.osw_p[k] for k in self.osw_p.keys() if grp_name==self.file_map[k]}
                    self.oscillo_fit(to_see)
                

    def pulse_segment(self, files, plot=True):
        lack_file = []
        for file in files:
            if plot:
                print(file)
            try:
                self.data_HL[file] = parse_data(file)
            except:
                print(f'{file} not found...')
                lack_file.append(file)
                #files.remove(file)
                continue
            try:
                s_idx, e_idx = self.data_HL[file]['trigger_sig'][2], self.data_HL[file]['trigger_sig'][-1]
            except:
                #input(file)
                #continue
                s_idx, e_idx = 122*5, -1
            sig_roi = self.data_HL[file]['signal_A1'][s_idx:e_idx]
            if len(sig_roi)==0:
                print('  ------------\n\n   No data here.\n\n  -------------')
            else:
                self.data_HL[file]['amp'], self.data_HL[file]['itvl'], self.data_HL[file]['base_idx'], self.data_HL[file]['peak_idx'] = re_seg_hd(sig_roi, plot=plot) # 20211112 re_seg_hd changed to v2
                self.data_HL[file]['base_idx'] += s_idx
                self.data_HL[file]['peak_idx'] += s_idx
        
        for file in lack_file:
            files.remove(file)

    def oscillo_extract(self, files): #, pressure_bound='DIA'
        for file in files:
            try:
                amp = np.array(self.data_HL[file]['amp'])
            except:
                print("Miss whole data: %s" %file)
                continue
            if 'ppg' in file:
                break
            base_idx = np.array(self.data_HL[file]['base_idx'])
            amp, base_idx = amp[amp>=amp_thrd], base_idx[amp>=amp_thrd]
            pressure_sig = self.data_HL[file]['signal_A0']    
            pressure_base = pressure_sig[base_idx]     
            try:
                #pressure_lim = data_HL[file]['DIA(m)'] # 20210630 change to use the following 2 lines
                turn_position = detect_turn(pressure_base, np.log(amp), debug=False)
                #print(f"Turn_position: {turn_position}")
                pressure_lim = pressure_base[turn_position]
            except:
                print("Miss pressure data: %s" %file)
                continue
                # print("Turning point not found")
                # pressure_lim = 0
            pressure_lim = int(pressure_lim)
            if pressure_lim==0:
                pressure_lim = 100
                print("Pressure zero value: %s" %file)
            x_data = pressure_base[pressure_base<pressure_lim]
            y_data = np.log(amp[pressure_base<pressure_lim])

            self.osw_p[file] = x_data, y_data
        
    def oscillo_fit(self, test_osw_p):
        gather_x, gather_y = [], []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for k,v in test_osw_p.items():
            sys, dia, hr = (self.data_HL[k][item] for item in ['SYS(d)', 'DIA(d)', 'Pulse(d)'])
            plt.scatter(v[0], v[1], label=k.split('/')[1][:-4]+f' {sys}_{dia}_{hr}', marker='x')
            gather_x += v[0].tolist()
            gather_y += v[1].tolist()

        full_result = polyfit(gather_x, gather_y, 1, full=True) 
        b, m = full_result[0]
        residual = full_result[1][0]
        gather_x = np.array(gather_x)
        plt.plot(gather_x, gather_x*m+b)
        plt.text(0.1, 0.85,'b = %.3f\nc = %.3f\nRMSE = %.3f'%(m,b,residual), ha='left', va='center', transform=ax.transAxes)
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.xlabel(r'$P_{ext}$')
        plt.ylabel('log(OWM)')
        title = self.file_map[k]#k.split('/')[1][:-6]
        plt.title(title)
        plt.grid()
        plt.show()   
        self.personal_param[title] = {'b':m, 'c':b, 'RMSE':residual}

# for removing initial outliers
PRESSURE_RATIO = 1.3
AMP_RATIO = 1.2


def derivative(x, y, back=False, fit=False):
    #f = interpolate.interp1d(x, y, kind='cubic')
    #y = f(x)
    if fit:
        a0, a1, a2, a3 = polyfit(x, y, 3)
        y = [a0+a1*xi+a2*xi**2+a3*xi**3 for xi in x]
    y_range = max(y)-min(y)
    x_range = max(x)-min(x)
    delta_y = [(i2-i1)/y_range for i1, i2 in zip(y[:-1], y[1:])]
    delta_x = [(i2-i1)/x_range for i1, i2 in zip(x[:-1], x[1:])]
    if back:
        return [dy/dx for dx, dy in zip(delta_x, delta_y)] + [0]
    return [0] + [dy/dx for dx, dy in zip(delta_x, delta_y)]

START_IDX = 6
def detect_turn(x, y, debug=False, fit=False):
    # starting detection point
    
    first_order = derivative(x, y, fit=fit)
    second_order = derivative(x, first_order, back=True)
    #print(first_order)
    #print(second_order)
    if debug:
        plt.plot(x[5:], y[5:], label='origin')
        plt.plot(x[5:], first_order[5:], label='1st')
        plt.plot(x[5:], second_order[5:], label='2nd')
        plt.legend()
        plt.show()
        print('--------------'*2)
        for n in range(len(x)):
            print(n, x[n], first_order[n])


    check_all = sorted(zip(range(5,len(y)), first_order[START_IDX:], second_order[START_IDX:]), \
                key=lambda x: x[2], reverse=True)
    if debug:
        print(check_all[::-1])
    check_one = check_all.pop()
    # Determine peak threshold for "fast" and "slow" speed
    SUM_THRD = -4 if np.mean(x[1:]-x[:-1])>2.5 else -1.5
    while not is_peak(x, first_order, second_order, check_one[0], SUM_THRD, debug):
        if debug:
            input(check_one[0])
        if len(check_all)==0:
            check_one = -1
            break
        check_one = check_all.pop()

    if check_one==-1:
        # 重複找，每次找不到就放寬threshold
        repeat = 1
        while check_one==-1:
            check_one = detect_turn_old(x, y, repeat)
            repeat += 1
            if repeat>5:
                break
        return check_one
    #   return detect_turn_old(x, y)

    return check_one[0]
    #return i-2 # think about it
def detect_turn_old(x, y, scale):
    TURNING_THRD = 0.5*scale
    y1 = derivative(x, y)
    y2 = derivative(x, y1, back=True)
    for i in range(len(y2)-2):
        if i > START_IDX:
            if y2[i]<TURNING_THRD: #and second_order[i+1]<TURNING_THRD
                if y1[i]>TURNING_THRD and -TURNING_THRD<y1[i+1]<2*TURNING_THRD and y1[i+2]<TURNING_THRD:
                    #print(x[i])#, y[i], first_order[i+1], second_order[i])
                    #print('Inside: detect_turn_old')
                    #input(f'y2:{y2[i:i+2]}')
                    if y2[i+2]>TURNING_THRD: # very strange, need to debug!!!
                        continue
                    #print('Got rom old')
                    return i
            #elif second_order[i]< (-2.5)*TURNING_THRD and first_order[i]>2*TURNING_THRD:
                #print(x[i], y[i], first_order[i+1], second_order[i])
            #    return i
    return -1

def is_rising(y1, i, check_pt=5, thrd=1):
    for k in range(i,i+check_pt):
        if y1[k]>thrd:
            return True
    return False


def is_peak(x, y1, y2, i, SUM_THRD, debug=False):
    sig1 = derivative(x, y2)   
    if i>=len(x)-7: # we'll keep the controller do the job. Get enough pulses after candidate found
        return False
    if sum(y1[i+1:i+3])>1.5 or sum(y1[i+1:i+3])<SUM_THRD or is_rising(y1,i+3) or y1[i-1]<0 or y1[i-2]<0: 
        return False
    if sig1[i]<0 and sig1[i+1]>0:
        #print('Got from new')
        return True
    return False

def is_contain(items, name):
    for item in items:
        if item in name:
            return True
    return False