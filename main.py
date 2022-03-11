import itertools
from scipy.special import comb
import numpy as np
import cmaest
import fuction_lib as fl
import time
import random

t1 = time.time()
# 回転角度のリスト [deg]
# ここでは回転前を0度、回転後を60度とする
# rot_angles = [0, 80]
# inputの内、音声ファイルの数
n_file = 10
n_mic = 5
n_src = 2
# 音声ファイル名のリスト
# wav_files = [f"./input/speech{n+1}.wav" for n in range(n_src)]
# wav_files = [f"./input/speech{n+1}_b.wav" for n in range(n_src)]
# rot_wav_files = [f"./input/speech{n+1}_a.wav" for n in range(n_src)]
# 組み合わせの数を計算
# a = comb(n_file, n_src, exact = True)
# ランダムに番号を取る
# n_ran = np.random.randint(1,a+1)
# print(n_ran)
# ランダムな番号に対応する音源ファイルを取る
# t = 0
# for pair in itertools.combinations(np.arange(1,n_file+1), n_src):
#    t=t+1
#    if t == n_ran:
#        pair_token = pair
# print(pair_token)
# print(np.size(pair_token))
# [theta0_src,theta1_src]
# locs_src = [30, 120]


# 全組み合わせ(大量な実験)
sources = [f"./input/speech{n+1}.wav" for n in range(10)]
pair = itertools.combinations(np.arange(1, n_file+1), n_src)#音声ファイル1-10から、n_src個を取るの全ての組み合わせ
pair = random.sample(list(pair), 10) #すべての組み合わせから10種を取る出す
pair = [[1,2]]
#import pdb; pdb.set_trace()
#[theta0_src,theta1_src]

locs_source = [[0,10],[0,20],[0,30],[0,40]]

#rot_angles = [[0,30]]
rot_angles = [[0,30], [0,60], [0,90],[0,120]]

id_experiment = -1
n_freq_max, n_frame_max = 200, 200

#path2 = f"./csv/test{id_experiment}.csv"
#path2 = 'experiment_rt355ms.csv'
#path2のファイルを空にする,最初の一回だけ使う。
#fl.to_csv(path2)
for rot_angle, locs_src, pair_token in itertools.product(rot_angles, locs_source, pair):
    id_experiment = id_experiment + 1
    print("experiment id:", id_experiment)
    print("All parameters:", pair_token, locs_src, rot_angle)
    # import pdb; pdb.set_trace()
    # 角度走査幅
    step_angle = 1
    measured_rt60 = cmaest.generate_signal(
        n_mic, pair_token, locs_src, rot_angle
    )

    X1, X2, L, est_angle = cmaest.estimate_cma_angle(
        t1, rot_angle, step_angle, n_freq_max, n_frame_max, n_mic
    )
    print(L.shape)
    fl.plot_time_likelihood_real(
       L, step_angle, n_src, pair_token, locs_src, rot_angle[1], measured_rt60,n_mic,id_experiment
    )
    #fl.plot_freq_likelihood_real(
    #L, step_angle, n_src, pair_token, locs_src, rot_angle[1], measured_rt60
    #)
    #fl.plot_likelihood_real(
    #    L, step_angle, n_src, pair_token, locs_src, rot_angle[1], measured_rt60
    #)
    #path1 = f"./csv/experiment{id_experiment}.csv"
    #尤度関数値の情報を保存
    #fl.writeto_csv(L, path1, step_angle, rot_angle[1], est_angle)

    #実験ID & パラメータの保存
    #fl.experment_tocsv(path2,id_experiment,n_src,pair_token,locs_src,n_mic,rot_angle[1],est_angle)
    #fl.test(L)
    if id_experiment == 0:
        break