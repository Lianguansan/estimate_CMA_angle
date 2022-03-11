import numpy as np
import itertools
from scipy.special import comb
import fuction_lib as fl
from scipy.io import wavfile
import pyroomacoustics as pra
import os

def set_locs_micarray1(n_mic, radius, rot, center):
    r = radius  #
    deg = [k * 360 / n_mic + rot for k in range(n_mic)]
    print(deg)
    rad = [d / 180 * np.pi for d in deg]
    mic_x = [r * np.cos(t) for t in rad]
    mic_y = [r * np.sin(t) for t in rad]
    mic_locs = [
        [center[0] + mic_x[i], center[1] + mic_y[i], center[2]] for i in range(n_mic)
    ]
    mic_locs = np.array(mic_locs).T
    #print(type(mic_locs))
    #print(mic_locs)
    return mic_x, mic_y, mic_locs

def obsignal_simu(pair_token, theta0_src, theta1_src, n_src, rot_angles):
    #wav_files = [f"./input/speech{n}_b.wav" for n in pair_token]
    #rot_wav_files = [f"./input/speech{n}_a.wav" for n in pair_token]
    # 音声ファイルの読み込み
    # (2 ** 15 - 1) で割り算しているのは、データフォーマットをfloatにするためです
    #wavs = [wavfile.read(f)[1] / (2 ** 15 - 1) for f in wav_files]
    #rot_wavs = [wavfile.read(f)[1] / (2 ** 15 - 1) for f in rot_wav_files]
    #信号を10s時点で分割する
    wav_files = [f"./input/speech{n}.wav" for n in pair_token]
    wavs = [wavfile.read(f)[1][:wavfile.read(f)[0]*10] / (2 ** 15 - 1) for f in wav_files]
    rot_wavs = [ wavfile.read(f)[1][wavfile.read(f)[0]*10:] / (2 ** 15 - 1) for f in wav_files]
    #wavs = [wavfile.read(f)[1] / (2 ** 15 - 1) for f in wav_files]
    #rot_wavs = [wavfile.read(f)[1] / (2 ** 15 - 1) for f in wav_files]
    # マイクロホン数
    n_mic = 5
    # マイクアレイの半径 [m]
    radius = 0.05

    # 部屋の形状 [m]: (x, y, z)
    room_dim = np.array([6.0, 4.0, 3.5])

    # シミュレーションのパラメータ
    room_params = {
        # サンプリング周波数
        "fs": 16000,
        # 壁面の吸音率
        "absorption": 0.05,
        # 鏡像法の計算次数。0にすると残響なし、1以上にすると残響あり
        "max_order": 9,
    }

    # 鏡像法では部屋の中心にマイクや音源を配置するとノイズが発生する
    # これを避けるためにランダムな値を足してずらす
    np.random.seed(0)
    center = room_dim / 2 + np.random.rand(*room_dim.shape) * 0.001

    # 回転前と回転後の2つのシミュレーションを行う
    for rot in rot_angles:
        # 部屋の作成
        room = pra.ShoeBox(p=room_dim, **room_params)
        mic_x, mic_y, mic_locs = set_locs_micarray1(n_mic, radius, rot, center)
        # mic_locs = set_locs_micarray2(n_mic, radius, rot)
        # 円状マイクロホンアレイを配置
        room.add_microphone_array(pra.MicrophoneArray(mic_locs, room.fs))

        # 音源の位置
        # ここでは部屋の中心から2mの円周上に均等に配置している
        # これをいじって色々試してみるといいです
        r = np.minimum(*room_dim[:2]) * 0.4  #
        deg = [theta0_src + k * theta1_src for k in range(n_src)]
        rad = [d / 180 * np.pi for d in deg]
        src_x = [r * np.cos(t) for t in rad]
        src_y = [r * np.sin(t) for t in rad]
        #src_xy = [[center[0] + src_x[i], center[1] + src_y[i]] for i in range(n_src)]
        src_locs = [
            [center[0] + src_x[i], center[1] + src_y[i], center[2]] for i in range(n_src)
        ]

        # 音源の配置
        # for sl, a in zip(src_locs, wavs):
        #    room.add_source(sl, signal=a)
        if rot == 0:
            for sl, a in zip(src_locs, wavs):
                room.add_source(sl, signal=a)
        else:
            for sl, a in zip(src_locs, rot_wavs):
                room.add_source(sl, signal=a)
        # 音源、マイクの配置を描
        fig, ax = room.plot(img_order=0)
        output_dir = f"./output/"
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f"{output_dir}room_layout_rot{rot:03}.png")

        # 室内インパルス応答の畳み込み
        room.compute_rir()

        # 残響時間の測定（将来、残響ありで実験するときに使ってみてください）
        measured_rt60 = pra.experimental.measure_rt60(room.rir[0][0], room.fs)
        print("RT60 is approximately", measured_rt60)

        # 混合前の信号
        # shape: (source, microphone, data)
        premix = room.simulate(return_premix=True)
        premix /= np.max(np.abs(premix))  # 正規化

        # 混合音の保存
        for i in range(n_mic):
            wavfile.write(
                f"{output_dir}mic{i+1}_rot{rot:03}.wav", room.fs, room.mic_array.signals[i]
            )

# 回転角度のリスト [deg]
# ここでは回転前を0度、回転後を60度とする
rot_angles = [0, 80]

# inputの内、音声ファイルの数
n_file = 10
# 音源数
n_src = 3

# 音声ファイル名のリスト
# wav_files = [f"./input/speech{n+1}.wav" for n in range(n_src)]
# wav_files = [f"./input/speech{n+1}_b.wav" for n in range(n_src)]
# rot_wav_files = [f"./input/speech{n+1}_a.wav" for n in range(n_src)]
# 組み合わせの数を計算
a = comb(n_file, n_src, exact=True)
# ランダムに番号を取る
n_ran = np.random.randint(1, a + 1)
print(n_ran)
# ランダムな番号に対応する音源ファイルを取る
t = 0
for pair in itertools.combinations(np.arange(1, n_file + 1), n_src):
    t = t + 1
    if t == n_ran:
        pair_token = pair
print(pair_token)
print(np.size(pair_token))

theta0_src = 30
theta1_src = 120

obsignal_simu(pair_token, theta0_src, theta1_src, n_src, rot_angles)

# 全組み合わせ
# for pair in itertools.combinations(np.arange(1,n_file+1), n_src):
#    obsignal_simu(pair, theta0_src, theta1_src, n_src, rot_angles)

n_mic = 5
output_dir = f"./output/"
# ファイルを選択して、STFTする
X1 = np.zeros((5, 513, 319), dtype=np.complex)
for i in range(n_mic):
    freq_num = 1024
    overlap = 512
    rate, data = wavfile.read(f"{output_dir}mic{i + 1}_rot{rot_angles[0]:03}.wav")
    # data = np.hstack([data, np.zeros(rate * 10 - np.size(data))])
    X1[i, :, :] = fl.my_STFT1(data, freq_num, overlap)

# import pdb;pdb.set_trace()

X2 = np.zeros((5, 513, 314), dtype=np.complex)
for i in range(n_mic):
    freq_num = 1024
    overlap = 512
    rate, data = wavfile.read(f"{output_dir}mic{i + 1}_rot{rot_angles[1]:03}.wav")
    if np.size(data) < rate * 10:
        data = np.hstack([data, np.zeros(rate * 10 - np.size(data))])
    else:
        data = data[: 10 * rate]
    X2[i, :, :] = fl.my_STFT1(data, freq_num, overlap)

# 角度推定
a = 5
chNum = 5
framNum = 2
freqNum = 513
# framNum = 501
# freqNum = 326
# L = fl.IndAngle_self(X1, X2, a, chNum, framNum, freqNum)
L = fl.IndAngle_signal_stft(X1, X2, a, chNum, framNum, freqNum)
fl.plot_freq_likelihood(L, a, chNum, framNum, freqNum)
fl.plot_likelihood(L, a, chNum, framNum, freqNum, rot_angles[1])
