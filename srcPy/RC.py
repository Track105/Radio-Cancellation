import numpy as np
from scipy.io import wavfile as wf
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

filter_ords = [16, 32, 64, 128, 256]
alphas = [0.001, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
# command_path = '../FastTestFiles/command.wav'
# music_path = '../FastTestFiles/radio_music.wav'
# combined_path = '../FastTestFiles/combined.wav'
# error_path = '../FastTestFiles/error_nlms.wav'
# path_music = "../TestFiles/music/"
# path_command = "../TestFiles/commands/"
# path_error = "../TestFiles/errors/"
# path_combined = "../TestFiles/combined/"

music_path = "../TestFilesRC/radio2_5min.wav"
music_command_filtered_path = "../TestFilesRC/combined_echo.wav"
recovered_command_path = "../TestFilesRC/voice.wav"

class Constants:
    
    delta = 0.00000001
    lamb = 0.994
    sr = 16000
    k = 2
    MVR = 0
    alpha = None
    filter_lengh = None      
    
def nlms(x, d, L, alpha, delta, lamb):
    
    N = len(x)
    e = np.zeros(N)
    y_est = np.zeros(N)
    h_est = np.zeros(L)
    xn = np.zeros(L)
    rem = 0
    sigma_m = 0 
    csi = 0
    pwr = 0

    for n in range(0, N):
        pwr = pwr + x[n] * x[n] - xn[L - 1] * xn[L - 1]
        xn = np.concatenate(([x[n]], xn[0:L - 1]))
        y_est[n] = np.dot(xn, h_est)
        e[n] = d[n] - y_est[n]
        
        rem = lamb * rem + (1 - lamb) * e[n] * x[n]
        sigma_m = lamb * sigma_m + (1 - lamb) * x[n] * x[n]
                
        csi = 1 if sigma_m == 0 else 1 - rem / sigma_m
                
        step = alpha / (pwr + delta)
        if (n >= 16000 and csi > 0.92 and csi <= 1.02) or n < 16000:
            h_est = h_est + step * e[n] * xn
    
    return -e

def nlms_rc(x, d, L, alpha, delta):
    
    N = len(x)
    e = np.zeros(N)
    y_est = np.zeros(N)
    m_est = np.zeros(N)
    voice = np.zeros(N)
    h_est = np.zeros(L)
    xn = np.zeros(L)
    mn = np.zeros(L)
    pwr = 0

    for n in tqdm(range(0, N)):
        pwr = pwr + x[n] * x[n] - xn[L - 1] * xn[L - 1]
        
        xn = np.concatenate(([x[n]], xn[0:L - 1]))    
        y_est[n] = np.dot(xn, h_est)
        e[n] = d[n] - y_est[n]
               
        mn = np.concatenate(([x[n]], mn[0:L - 1]))     
        m_est[n] = np.dot(mn, h_est)
        voice[n] = d[n] - m_est[n]
                
        step = alpha / (pwr + delta)
        if n < 32000 or (n >= 32000 and np.abs(voice[n]) < 0.0005):
            h_est = h_est + step * e[n] * xn
            
    return -e, voice

def read_signals(command_path, music_path):
    
    sr, command = wf.read(command_path)
    sr, music = wf.read(music_path)
    if abs(max(command)) > 1:
        command = command.copy() / 2**15
    if abs(max(music)) > 1:
        music = music.copy() / 2**15
    return command, music

def create_far_end_signal(command, music, k, MVR):
    
    start = len(command)
    Px = np.sqrt(np.sum(command * command) / len(command))
    Py = Px / np.power(10, MVR / 10) / np.sqrt(np.sum(music * music) / len(music))
    music = music.copy() * Py
    combined = music.copy()
    combined[k*start:k*start + len(command)] += command
    return combined, music

def find_best_coeffs(command, music, combined, delta, lamb, k):
    
    start = len(command)
    scores = {}

    command_corr = np.correlate(command, command)[0]
    
    for alpha in alphas:
        for filter_ord in filter_ords:
            e_nlms = nlms(x, music, filter_ord, alpha, delta, lamb)
            corr = np.correlate(command, e_nlms[k*start:(k+1)*start])[0] / command_corr
            if corr > 1 or corr < 0:
                corr = 0
            energy = np.sum(e_nlms[k*start:(k+1)*start] ** 2) / np.sum(e_nlms ** 2)
            scores[(alpha, filter_ord)] = (corr, energy)
            print("alpha: {}, filter_ord: {}, corr: {}, energy: {}".format(alpha, filter_ord, corr, energy))
    
    list_scores = list(scores.items())
    list_scores.sort(key=lambda x: -(3 * x[1][0] + 2 * x[1][1]) / 5)
    
    alpha = list_scores[0][0][0]
    filter_length = list_scores[0][0][1]
    
    return list_scores, alpha, filter_length

def compute_all_files_nlms(path_music, path_command, path_combined, path_error, Constants):
    
    music_filenames = []
    command_filenames = []
    for root, dirnames, filenames in os.walk(path_music):
        music_filenames = filenames
    for root, dirnames, filenames in os.walk(path_command):
        command_filenames = filenames
        
    for i in range(50):
        for MVR in [-5, 0, 5]:
            command, music = read_signals(path_command + command_filenames[i], path_music + music_filenames[i])
            x, music = create_far_end_signal(command, music, Constants.k, MVR)
            wf.write(filename=path_combined + str(i + 1) + "_MVR_" + str(MVR) + ".wav", rate=Constants.sr, data=x)
            list_scores, Constants.alpha, Constants.filter_length = find_best_coeffs(command, music, x, Constants.delta, Constants.lamb, Constants.k)
            
            e_nlms = nlms(x, music, Constants.filter_length, Constants.alpha, Constants.delta, Constants.lamb)
            wf.write(filename=path_error + str(i + 1) + "_MVR_" + str(MVR) + "_alpha_" + str(Constants.alpha) + "_ord_" + str(Constants.filter_length) + ".wav", rate=Constants.sr, data=e_nlms)

# command, music = read_signals(command_path, music_path)
# start = len(command)
# x, music = create_far_end_signal(command, music, Constants.k, Constants.MVR)
# wf.write(filename=combined_path, rate=Constants.sr, data=x)
# list_scores, Constants.alpha, Constants.filter_length = find_best_coeffs(command, music, x, Constants.delta, Constants.lamb, Constants.k)

# sr, x = wf.read(filename="/home/apopa/Downloads/radio2_5min.wav")
# sr, x_filt = wf.read(filename="/home/apopa/Downloads/combined_echo.wav")

# e_nlms = nlms(x, music, 64, 0.125, Constants.delta, Constants.lamb)
# wf.write(filename=error_path, rate=Constants.sr, data=e_nlms)

sr, x = wf.read(filename=music_path)
sr, x_filt = wf.read(filename=music_command_filtered_path)

e_nlms_rc, voice = nlms_rc(x, x_filt, 64, 0.6, 0.00000001)
wf.write(filename=recovered_command_path, rate=Constants.sr, data=voice)
















