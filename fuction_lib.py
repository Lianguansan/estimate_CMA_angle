import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy.io import wavfile
import pandas as pd
import time
import seaborn as sns



def Loglikelihood(V, x, n):
    # 0.5 * (x - mu).T * V.I * (x - mu)
    P = (
        -n * np.log(np.pi)
        - np.log(np.linalg.det(V))
        - np.conj(x).T.dot(np.linalg.inv(V)).dot(x)
    )
    # P = np.exp(P)
    return P.real  # positive log)

def log_likelihood(V, x):  # size of rotX2:(360//step,f,m,t), n:n_mic
    L = (
        -V.shape[1] * np.log(np.pi)  # size of V: (n_freq, n_mic, n_mic)
        - np.log(np.linalg.det(V)).reshape(V.shape[0], 1)  # det(V) : (n_freq,)
        - np.diagonal(
            np.conj(np.transpose(x, (0, 1, 3, 2))) @ np.linalg.inv(V) @ x,
            offset=0,
            axis1=2,
            axis2=3,
        )
    )
    # (theta, f, t,m) * (f,m,m)*(theta,f,m,t) = (theta, f, t,t)
    # diagonal :対角成分を取る: reduce demension from 4 to 3
    # reduce shape of L from (theta,f,t,t) to (theta,f,t)
    # L.shape = -(theta,f,t) - (f,1) - float
    return L.real  # L:(theta, f, t)

def loglikelihood(V, x):# size of rotX2:(360//step,f,m,t)
    print('x2.shape',x.shape)
    x = np.transpose(x,(0,1,3,2)) # change shape of x to  (360//step,f,t,m)
    x = x[:,:,:,:,np.newaxis] # shape of x to (360//step,f,t,m,1)
    V = V[np.newaxis, :, np.newaxis, :, :] # V to (1,f,1,m,m)
    L = (
        -V.shape[3] * np.log(np.pi)  # size of V: (n_freq, n_mic, n_mic)
        - np.log(np.linalg.det(V)).reshape(V.shape[1], 1)  # det(V) : (n_freq,)
        - (np.conj(np.transpose(x, (0, 1, 2, 4, 3))) @ np.linalg.inv(V) @ x).reshape(x.shape[0],x.shape[1],x.shape[2])
    )
    return L.real # L:(theta, f, t)

def CalcRotationMatrix(M, rotAngleDeg):
    rotAngleRad = rotAngleDeg / 180 * np.pi
    delay = rotAngleRad / (2 * np.pi / M)  # sptlSubSampleShift

    x = np.linspace(0, M - 1, M)
    y = np.linspace(0, M - 1, M)
    n, m = np.meshgrid(x, y)
    L = n - m - delay

    if M % 2 == 0:
        U = (1 + np.exp(-1j * np.pi * L)) / M + np.cos(
            L * np.pi * (M + 2) / (2 * M)
        ) * (np.sinc(L / 2) / np.sinc(L / M))
    else:
        U = 1 / M + ((M - 1) / M) * np.cos(L * np.pi * (M + 1) / (2 * M)) * (
            np.sinc(L * (M - 1) / (2 * M)) / np.sinc(L / M)
        )

    return U


def calc_rotation_matrix(M, step_angle):  # ベクトル演算用
    angle_deg = np.arange(-180, 180, step_angle)  # [0,360//step)
    # U = np.zeros((360 // step_angle, M, M))
    # L = np.zeros((360//step_angle,M,M))
    # angle_deg = angle_index * step_angle - 180    #[-180, 180): size of: 360//step
    angle_rad = angle_deg / 180 * np.pi  # [-pi, pi)
    delay = angle_rad / (2 * np.pi / M)  # sptlSubSampleShift [0,1,2]

    x = np.linspace(0, M - 1, M)
    y = np.linspace(0, M - 1, M)
    n, m = np.meshgrid(x, y)  # n,m :size(n_mic,n_mic)
    # print(n,m)
    delay = delay[:, np.newaxis, np.newaxis]
    L = n - m - delay  # shape of L :(360//step, M, M)
    # print('L.shape:',L)

    if M % 2 == 0:
        U = (1 + np.exp(-1j * np.pi * L)) / M + np.cos(
            L * np.pi * (M + 2) / (2 * M)
        ) * (np.sinc(L / 2) / np.sinc(L / M))
    else:
        U = 1 / M + ((M - 1) / M) * np.cos(L * np.pi * (M + 1) / (2 * M)) * (
            np.sinc(L * (M - 1) / (2 * M)) / np.sinc(L / M)
        )

    return U


def pltspec(X, freqNum, frameNum, chNum, sfreq):
    Hz = np.linspace(0, sfreq / 2, freqNum)
    t = np.linspace(0, freqNum * frameNum * chNum / sfreq, frameNum)
    Y = np.zeros(freqNum * frameNum * chNum, dtype=np.complex).reshape(
        chNum, frameNum, freqNum
    )
    for i in range(chNum):
        for n in range(frameNum):
            for m in range(freqNum):
                Y[i, n, m] = 10 * np.log10(abs(X[i, n, m] + 10 ** -10) ** 2)
                # Y[i, n, m] = np.abs(X[i, n, m])
        plt.subplot(3, 2, i + 1)
        plt.pcolormesh(t, Hz, np.abs(Y[i, :, :]).T)
        # str = "Channel" + i
        plt.title("Channel " + str(i + 1))
        plt.xlabel("t [s]")
        plt.ylabel("f [Hz]")
        # plt.colorbar()
        h = plt.colorbar()
        h.set_label("Power (dB)")

    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    return


# 共分散行列
def Cov1(Nt, X):  # (n_mic, n_frame)
    # I = np.eye((5), dtype=np.complex)
    V = 1 / Nt * X.dot(np.conj(X).T)  # + 0.00000001 * I
    return V.real


def cov1(X):  # ベクトル演算用
    # size of X: (n_freq, n_mic, n_frame)
    V = 1 / X.shape[2] * X @ np.transpose(np.conj(X), (0, 2, 1))
    return V  # shape of V : (n_freq, n_mic, n_mic)


def Cov2(X):
    # I = np.eye((5), dtype=np.complex)
    V = np.cov(X)  # + 0.00000001 * I
    return V


def my_STFT1(data, nperseg, overlap):
    fs = 16000
    # x = np.frombuffer(data, "int16")

    # 计算并绘制STFT的大小
    f, t, Zxx = signal.stft(data, fs, nperseg=nperseg, noverlap=overlap)

    # plt.pcolormesh(t,f,np.abs(Zxx))
    # plt.show()
    return Zxx


def IndAngle_signal_stft(X1, X2, step_angle, n_freq_max=None, n_frame_max=None):
    n_mic, n_freq, n_frame = X2.shape
    if n_freq_max is not None:
        n_freq = n_freq_max
    if n_freq_max is not None:
        n_frame = n_frame_max
    L = np.zeros((360 // step_angle, n_frame, n_freq))
    for theta in range(-180, 180, step_angle):
        U = CalcRotationMatrix(n_mic, theta)
        for indFreq in range(0, n_freq):  # 回転前の信号Xr,回転後の信号Xi,回転行列をかけた信号rotXi
            Xr = np.squeeze(X1[:, indFreq, :])  # ある周波数ビンの回転前の信号の全フレームを使って、共分散行列を推定
            sigma = Cov1(X1.shape[2], Xr)
            # sigma = Cov2(Xr)
            Xi = np.squeeze(X2[:, indFreq, :])
            rotXi = np.dot(U, Xi)
            for indFra in range(0, n_frame):
                rotxi = np.squeeze(rotXi[:, indFra])
                L[(theta + 180) // step_angle, indFra, indFreq] = Loglikelihood(
                    sigma, rotxi, n_mic
                )
    return L


def ind_angle_signal_stft(
    t1, X1, X2, step_angle, n_freq_max=None, n_frame_max=None
):  # ベクトル演算用
    n_mic, n_freq, n_frame = X2.shape
    print('X2.n_frame',n_frame)
    if n_freq_max is not None:
        n_freq = n_freq_max
    if n_frame_max is not None:
        n_frame = n_frame_max
    # L = np.zeros((360 // step_angle , n_frame, n_freq))
    U = calc_rotation_matrix(
        n_mic, step_angle
    )  # size of U: (360//step_angle, n_mic, n_mic)
    U = U[:, np.newaxis, :, :]  # (angle_index,1,m,m)
    X1 = np.transpose(
        X1[:, :n_freq, :n_frame], (1, 0, 2)
    )  # change size of X1 from (m,f,t) to (f,m,t)
    print('X1',X1.shape)
    covariance = cov1(X1)  # size of cov: (n_freq, n_mic, n_mic)
    X2 = np.transpose(
        X2[:, :n_freq, :], (1, 0, 2) #batch処理時 n_frame(=200)を追加,online処理は削除
    )  # change size of X2 from (m,f,t) to (f,m,t)
    X2 = X2[np.newaxis, :, :, :]  # change size from (f,m,t) to (1,f,m,t)
    rotX2 = U @ X2  # size of rotX2: (360//step,f,m,t)
    print('n_frame',n_frame_max)
    print("X2",X2.shape)
    print('rotX2',rotX2.shape)
    L = loglikelihood(covariance, rotX2)
    L = np.transpose(L, (0, 2, 1))  # change size from (theta,f,t) to (theta,t,f)
    print("L:",L.shape)
    l1 = L.mean(axis=(1, 2))
    # print(np.argmax(l1))
    #max_index1 = np.argmax(l1[::-1]) + 1 # [-180,179)なので、-30°を30°にするとき、
    max_index = np.argmax(l1)
    #max_angle1 = max_index1 * step_angle - 180
    max_angle = -(max_index * step_angle - 180)
    print("全フレームを使った推定角度: ", max_angle)
    t2 = time.time()
    elapsed_time = t2 - t1
    print("処理時間: ", "{:.1f}".format(elapsed_time), "秒")
    return L, max_angle


def estimate_cma_angle(rot_angles):
    n_mic = 5
    output_dir = f"./output/"
    X1 = []
    X2 = []
    fft_num = 1024
    overlap = 512
    for i in range(n_mic):
        rate, data = wavfile.read(f"{output_dir}mic{i + 1}_rot{rot_angles[0]:03}.wav")
        # data = np.hstack([data, np.zeros(rate * 10 - np.size(data))])
        X1.append(fl.my_STFT1(data, fft_num, overlap))
        rate, data = wavfile.read(f"{output_dir}mic{i + 1}_rot{rot_angles[1]:03}.wav")
        X2.append(fl.my_STFT1(data, fft_num, overlap))
    X1 = np.array(X1)
    X2 = np.array(X2)
    # 角度推定
    step_angle = 10
    n_mic, n_freq, n_frame = X1.shape
    # L = fl.IndAngle_self(X1, X2, a, chNum, framNum, freqNum)
    # L = fl.IndAngle_signal_stft(X1, X2, step_angle, n_mic, n_frame, n_freq)


def plot_freq_likelihood_real(
    L, step_angle, n_src, pair_token, locs_src, rot_angle, measured_rt60,fs = 16000, n_framemax = 513
):

    n_theta, n_frame, n_freq = L.shape
    print('L.shape',L.shape)
    l = L.mean(axis=1)
    # print(np.argmax(l, axis=0))
    indFre = [f * fs / 2 / n_framemax /1000 for f in np.arange(n_freq)]
    # indFre = np.arange(freqNum)
    theta = np.arange(-180, 180, step_angle)
    #cm = plt.cm.get_cmap('plasma')
    #plt.pcolormesh(l[::-1].T)
    plt.pcolormesh(theta, indFre, l[::-1].T, vmin = -20, vmax = 60) #2 sources
    #plt.pcolormesh(theta, indFre, l[::-1].T)
    #plt.pcolormesh(theta, indFre, np.log(np.abs(l[::-1].T)))
    # , cmap = "inferno"
    h = plt.colorbar()
    #h.ax.set_yticklabels([ "-7k","-6k", "-5k", "-4k", "-3k", "-2k", "-1k", "0"])
    h.ax.tick_params(labelsize = 16)
    #h.set_label('Log likelihood',fontsize = 18)
    plt.xlabel("Angle [deg]",fontsize = 18)
    plt.ylabel("Frequency [kHz] ",fontsize = 18)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=20)
    # ytick = np.array(["0", "1", "2", "3", "4","5","6","7"])
    # locs = np.linspace(0, 8000, 9)
    # plt.yticks(locs, ytick)
    # 科学計数法
    #ax = plt.gca()
    #ax.ticklabel_format(style="sci", scilimits=(-1, 1), axis="y")
    loglike_dir = f"./log-likelihood_fig/"
    #plt.savefig(
    #    f"{loglike_dir}_{n_src}_{pair_token}_{locs_src}_{rot_angle}_rt{measured_rt60:03}freq.pdf",
    #    bbox_inches="tight",
    #)
    #plt.axvline(rot_angle, c="r", linestyle="--", marker="s")
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_freq_likelihood(
    L, step_angle, n_src, pair_token, locs_src, rot_angle, measured_rt60,id
):
    n_theta, n_frame, n_freq = L.shape
    print(L.shape)
    l = L.mean(axis=1)
    # print(np.argmax(l, axis=0))
    indFre = [f * 8000 / 513 for f in np.arange(n_freq)]
    # indFre = np.arange(freqNum)
    theta = np.arange(-180, 180, step_angle)
    # plt.pcolormesh(theta, indFre, l[::-1].T,rasterized=True,vmin = -8000, vmax = 0)
    if n_src == 2:
        plt.gcf().text(0.005, 0.4, "Estimated angle [deg]", rotation=90)
        plt.subplot(2,2,2*id+1)
    elif n_src == 3:
        plt.subplot(2, 2, 2*id + 2)
    #plt.pcolormesh(theta, indFre, l[::-1].T)
    plt.pcolormesh(theta, indFre, l[::-1].T,rasterized=True,vmin = -8000, vmax = 0)
    plt.xlabel("angle [deg]")
    plt.ylabel("frequency [kHz] ")
    # 科学計数法
    #ax = plt.gca()
    #ax.ticklabel_format(style="sci", scilimits=(-1, 1), axis="y")
    if n_src == 3 and id ==1:
        plt.tight_layout()
        plt.show()
        plt.close()


def plot_likelihood(L, step_angle, n_src, pair_token, locs_src, rot_angle, measured_rt60,id):
    l1 = L.mean(axis=(1, 2))
    theta = np.arange(-180, 180, step_angle)
    degs_src = [locs_src[0] + k * locs_src[1] for k in range(len(pair_token))]
    if n_src == 2:
        #plt.gcf().text(0.005, 0.4, "Estimated angle [deg]", rotation=90)
        plt.subplot(2,2,2*id+1)
        plt.ylabel("log-likelihood")
    elif n_src == 3:
        plt.subplot(2, 2, 2*id+2)
    plt.plot(theta, l1[::-1])
    plt.axvline(rot_angle, c="r", linestyle="--", marker="s")
    plt.title('location of sources: '+str(degs_src))
    plt.xlabel("angle [deg]")
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    #ax = plt.gca()
    #ax.ticklabel_format(style="sci", scilimits=(-1, 1), axis="y")
    # plt.rcParams["font.size"] = 15
    loglike_dir = f"./log-likelihood_fig/"
    #plt.savefig(
    #    f"{loglike_dir}_{n_src}_{pair_token}_{locs_src}_{rot_angle}_rt{measured_rt60:03}.pdf",
    #    bbox_inches="tight",
    #)
    print('n_src,id:',n_src,id)
    if id == 1:
        plt.xlabel("angle [deg]")
    if n_src == 3 and id == 1:
        plt.tight_layout()
        plt.show()
        #plt.close()

def plot_likelihood_real(L, step_angle, n_src, pair_token, locs_src, rot_angle, measured_rt60):
    l1 = L.mean(axis=(1, 2))
    theta = np.arange(-180, 180, step_angle)
    plt.plot(theta, l1[::-1])
    plt.axvline(rot_angle, c="r", linestyle="--", marker="s")

    plt.xlabel("Angle [deg]",fontsize = 18)
    plt.ylabel("Log-likelihood",fontsize =  18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=20)
    #ax = plt.gca()
    #ax.ticklabel_format(style="sci", scilimits=(-1, 1), axis="y")
    #plt.rcParams["font.size"] = 18
    #loglike_dir = f"./log-likelihood_fig/"
    # plt.savefig(
    #    f"{loglike_dir}_{n_src}_{pair_token}_{locs_src}_{rot_angle}_rt{measured_rt60:03}.pdf",
    #    bbox_inches="tight",
    # )
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_time_likelihood(L, step_angle, n_src, pair_token, locs_src, rot_angle, measured_rt60, n_mic,id,fs = 16000,overlap = 512):
    import re
    #L = np.transpose(L, (1, 2, 0))  # change shape from (theta,t,f) to (t,f,theta)
    n_angledex, n_frame, n_freq = L.shape
    print(L.shape)
    degs_src = [locs_src[0] + k * locs_src[1] for k in range(len(pair_token))]
    l1 = L.mean(axis=2) #shape of l1 :(theta,t)
    print(l1.shape)
    l1 = moving_average(l1, step=5)  # step点移動平均フィルタをかける
    # print(np.argmax(l1))
    max_index = np.argmax(l1,axis = 0)
    max_angle = -(max_index * step_angle - 180)
    print(max_angle.shape)
    print('id',id)
    fs = 16000
    #max_angle = max_angle[::-1]
    #print("1フレームを使った推定角度: ", max_angle)
    #max_angle = moving_average(max_angle, step=5) # step点移動平均フィルタをかける
    t = np.linspace(0, overlap * n_frame / fs, n_frame)
    plt.gcf().text(0.005, 0.3, "Estimated angle [deg]", rotation=90,fontsize = 14)
    if n_src == 2:
        plt.subplot(2,2,2*id+1)
    elif n_src == 3:
        plt.subplot(2, 2, 2*id + 2)
    plt.scatter(t, max_angle,s = 3)
    #plt.ylabel(str(degs_src))
    plt.xlabel("Time [s]",fontsize= 13)
    #degs_src = '[' + '°, '.join(re.findall("(?<=\[).+?(?=\])", str(degs_src))[0].split(', ')) + '°]'
    if n_src == 2:
        degs_src = '['+ str(degs_src[0])+'°, '+str(degs_src[1]) + '°]'
    elif n_src == 3:
        degs_src = '[' + str(degs_src[0]) + '°, ' + str(degs_src[1]) + '°, ' + str(degs_src[2]) + '°]'
    #plt.title('Location of sources: '+degs_src,fontsize =13)
    #plt.title('Location of sources: ' + degs_src, fontsize=13)
    plt.title('Number of sources: '+str(n_src)+', Angle between sources: '+str(locs_src[1])+' [deg]',fontsize = 12)

    plt.ylim(-10,40)
    plt.xlim(6,20)
    plt.xticks([6,10,20])
    plt.yticks([0, 30])
    plt.axhline(rot_angle, xmin=0.286, xmax=1, c="r", linestyle="--", marker="s")
    plt.axhline(0, xmin=0, xmax=0.286, c="r", linestyle="--", marker="s")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    if n_src == 3 and id == 1:
        #plt.tight_layout()
        plt.show()
        plt.close()
    #plt.close()
    #import pdb; pdb.set_trace()
    #plt.axvline(rot_angle, c="r", linestyle="--", marker="s")
    #ax = plt.gca()
    #ax.ticklabel_format(style="sci", scilimits=(-1, 1), axis="y")
    # plt.rcParams["font.size"] = 15
    #loglike_dir = f"./log-likelihood_fig/"
    #plt.savefig(
        #f"{loglike_dir}_{n_src}_{pair_token}_{locs_src}_{rot_angle}_rt{measured_rt60:03}.pdf",
        #bbox_inches="tight",
    #)

def plot_time_likelihood_real(L, step_angle, n_src, pair_token, locs_src, rot_angle, measured_rt60,n_mic,id,fs = 16000,overlap = 512):
    #L = np.transpose(L, (1, 2, 0))  # change shape from (theta,t,f) to (t,f,theta)
    n_angledex, n_frame, n_freq = L.shape
    print(L.shape)
    degs_src = [locs_src[0] + k * locs_src[1] for k in range(len(pair_token))]
    l1 = L.mean(axis=2) #shape of l1 :(theta,t)
    print(l1.shape)
    l1 = moving_average(l1, step=5)  # step点移動平均フィルタをかける
    # print(np.argmax(l1))
    max_index = np.argmax(l1,axis = 0)
    max_angle = -(max_index * step_angle - 180)
    print(max_angle.shape)
    print('id',id)
    fs = 16000
    #max_angle = max_angle[::-1]
    #print("1フレームを使った推定角度: ", max_angle)
    #max_angle = moving_average(max_angle, step=5) # step点移動平均フィルタをかける
    t = np.linspace(0, overlap * n_frame / fs, n_frame)
    #if n_src == 2:
    #    plt.gcf().text(0.005, 0.4, "Estimated angle [deg]", rotation=90)
    #    plt.subplot(3,2,2*id+1)
    #elif n_src == 3:
    ##    plt.subplot(3, 2, 2*id + 2)
    plt.scatter(t, max_angle,s = 5)
    plt.ylabel("Estimate [deg]",fontsize=15)
    plt.xlabel("Time [s]",fontsize=15)
    #plt.title('location of sources: '+str(degs_src))
    #plt.title('Number of sources: '+str(n_src)+', Angle between sources: '+str(locs_src[1])+' [deg]')
    plt.title('Number of sources = '+str(n_src),fontsize=16)

    plt.ylim(-6,40)
    plt.xlim(6,20)
    plt.yticks([-5,0,5,10,15,20,25,30,35,40])
    plt.axhline(rot_angle, xmin=0.286, xmax=1, c="r", linestyle="--", marker="s")
    plt.axhline(0, xmin=0, xmax=0.286, c="r", linestyle="--", marker="s")
    #if n_src == 3 and id == 2:
    plt.xticks(fontsize=15) #slide
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()
    plt.close()
    #import pdb; pdb.set_trace()
    #plt.axvline(rot_angle, c="r", linestyle="--", marker="s")
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    #ax = plt.gca()
    #ax.ticklabel_format(style="sci", scilimits=(-1, 1), axis="y")
    # plt.rcParams["font.size"] = 15
    #loglike_dir = f"./log-likelihood_fig/"
    #plt.savefig(
        #f"{loglike_dir}_{n_src}_{pair_token}_{locs_src}_{rot_angle}_rt{measured_rt60:03}.pdf",
        #bbox_inches="tight",
    #)

def boxplot(path):
    import re
    d = pd.read_csv(path)
    #print(d.head())
    #rot_angle = d.iloc[0,-2]
    rot_angle = -30 #-30,60,90,120,-120
    mask1 = d["rot_angle"]==rot_angle
    #locs_src_deg = "[0°, 25°, 50°]" #str型に変換してから、要素選択
    locs_src_deg = "[0, 25, 50, 75]"  # str型に変換してから、要素選択
    locs_src = "[0, 30]" #str型に変換してから、要素選択
    #mask_deg = d["locs_src"] == locs_src_deg
    #print('mask_deg',mask_deg)
    #d.loc[mask_deg, 'locs_src'] = "[0°, 25°, 50°, 75°]"
    #$for i, d1 in enumerate(d["locs_src"]):
     #   d.loc[i, "locs_src"] = '[' + '°, '.join(re.findall("(?<=\[).+?(?=\])", d1)[0].split(', ')) + '°]'
        #print('[' + '°, '.join(re.findall("(?<=\[).+?(?=\])", d1)[0].split(', ')) + '°]')
    #d[mask_deg]["locs_src"] = "[0°, 25°, 50°, 75°]"
    #print(type(locs_src))
    #print(type(d["locs_src"]))
    #d["locs_src"].astype('list')
    #d["locs_src"] = list(d["locs_src"])
    #print(type(d["locs_src"]))
    mask2 = d["locs_src"]==locs_src
    #print(mask2)
    #print(d.loc[mask1,"id"]) # d.loc[]はさらに行を指定できる、idの一行だけ出力
    n_src = 6
    mask_src = d['n_src'] == n_src
    #loc_src = '[0, 10]'
    #mask_loc = d["locs_src"] == loc_src,
    #print(mask_loc)
    print(mask_src)

    d.loc[d["locs_src"] == "[0, 10]", 'locs_src'] = "10°"
    d.loc[d["locs_src"] == "[0, 20]", 'locs_src'] = "20°"
    d.loc[d["locs_src"] == "[0, 30]", 'locs_src'] = "30°"
    d.loc[d["locs_src"] == "[0, 40]", 'locs_src'] = "40°"
    #print(d[mask2])
    #for i, d1 in enumerate(d["locs_src"]):
    #    d.loc[i, "locs_src"] = '[' + '°, '.join(re.findall("(?<=\[).+?(?=\])", d1)[0].split(', ')) + '°]'
        # print('[' + '°, '.join(re.findall("(?<=\[).+?(?=\])", d1)[0].split(', ')) + '°]')
    #for i, d1 in enumerate(d["degs_src"]):
    ##    d.loc[i, "degs_src"] = '[' + '°, '.join(re.findall("(?<=\[).+?(?=\])", d1)[0].split(', ')) + '°]'
     #   print('[' + '°, '.join(re.findall("(?<=\[).+?(?=\])", d1)[0].split(', ')) + '°]')
    #sns.boxplot(x="locs_src", y="error", data=d[mask_src])
    #sns.boxplot(x="degs_src", y="error", data=d[mask_src])
    sns.boxplot(x="n_src", y="error", data=d)
    plt.xticks(fontsize = 13, rotation = 0)
    plt.yticks(fontsize=13)
    #sns.boxplot(x="rot_angle", y="estimated_angle", data=d[mask2])
    plt.axhline(0, c="r", linestyle="--", marker="s")
    #plt.ylim(0,179)
    plt.ylabel('Error [deg]',fontsize = 13)
    plt.xlabel('Number of sources',fontsize = 13)
    #plt.xlabel('Angle of sources [deg]',fontsize = 13)
    #plt.title("Number of sources = "+str(n_src),fontsize = 14)
    #plt.title(locs_src)
    plt.ylim(-10,10)
    plt.tight_layout()
    plt.show()
    plt.close()

def boxplot_oneframe(path):

    d = pd.read_csv(path)
    for t in np.arange(315):
        mask1 = d[d['time frame'] == t]['log-likelihood'].idxmax()
    #mask1 = (d['angle']==d['angle']).idxmax()
    #print(d.loc[mask1,"id"]) # d.loc[]はさらに行を指定できる、idの一行だけ出力
    #print(d[mask2])
        sns.swarmplot(x="rot angle", y="angle", data=d[mask1])
    #sns.boxplot(x="rot_angle", y="estimated_angle", data=d[mask2])
    #plt.axhline(rot_angle, c="r", linestyle="--", marker="s")
    #plt.ylim(0,179)
    plt.ylabel('error [deg]')
    plt.xlabel('location patterns of sources')
    #plt.ylim(-6,3)
    plt.show()
    plt.close()


def moving_average(x,step):
    # 移動平均フィルタ
    print('step',step)
    for i in range(step-1, x.shape[1]):  # xは0から
        x[:,i] = np.mean(x[:,i - step + 1:i +1],axis = 1)
    return x

def write_csv(L, path):
    # csvの書き出し

    import csv
    with open(path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["angle", "time frame", "freq", "log-likelihood"])
        writer.writerows(L)


def writeto_csv(L, path, step_angle, rot_angle, est_angle):
    n_theta, n_frame, n_freq = L.shape
    print(L.shape)
    # csvの書き出し
    # angle&t
    a = []
    t = []
    for i in range(-180, 180, step_angle):
        a.append([i for _ in range(n_frame * n_freq)])
        for i in range(n_frame):  # n_frame
            t.append([i for _ in range(n_freq)])
    # print('t:', np.shape(t))
    # print(type(t))
    a = np.array(a).reshape(n_theta * n_frame * n_freq)
    t = np.array(t).reshape(n_theta * n_frame * n_freq)
    # print(t)
    # freq
    f = np.array([np.arange(n_freq) for _ in range(n_theta * n_frame)]).reshape(
        n_theta * n_frame * n_freq
    )
    # log-likelihood
    l = L.reshape(n_theta * n_frame * n_freq)
    df = pd.DataFrame(
        {
            "angle": a,
            "time_frame": t,
            "freq": f,
            "log-likelihood": l,
            "rot_angle": rot_angle,
            "estimated_angle": est_angle,
        }
    )
    # print(df)
    df.to_csv(path, index=False)


def experment_tocsv(
    path, id_experiment, n_src, pair_token, locs_src, n_mic, rot_angle, est_angle
):
    degs_src = [locs_src[0] + k * locs_src[1] for k in range(len(pair_token))]
    d = pd.read_csv(path)
    print(d)
    #print(type(locs_src))
    df = pd.DataFrame(
        {
            "id": id_experiment,
            "n_src": n_src,
            "files_src": [pair_token],
            "locs_src": [locs_src],
            "degs_src": [degs_src],
            "n_mic": n_mic,
            "rot_angle": rot_angle,
            "estimated_angle": est_angle,
            "error": est_angle - rot_angle,
        }
    )
    d1 = d.append(df)  # , ignore_index = True
    print(df)
    print(d.append(df))
    d1.to_csv(path, index=False)


def to_csv(path):
    # タイトルだけの空のcsvファイルを作る
    d = pd.DataFrame(
        columns=[
            "id",
            "n_src",
            "files_src",
            "locs_src",
            "n_mic",
            "rot_angle",
            "estimated_angle",
        ]
    )
    d.to_csv(path, index=False)
