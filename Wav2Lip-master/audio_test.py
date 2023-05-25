import audio

import os
import audio
import numpy as np
import cv2
import librosa
import time

from hparams import hparams as hp

basePath = '/home/ps/yulj/PycharmProjects/pythonProject/Wav2Lip-GFPGAN'
audio_file = basePath + '/inputs/1.wav'

audio_file = '/home/ps/yulj/PycharmProjects/pythonProject/source/aiortc/examples/server/5a7bdba7-563a-44ce-95fa-ded35217310b1_2.wav'

face_file = basePath + '/outputs/result.mp4'


wav2lipFolderName = 'Wav2Lip-master'
wav2lipPath = basePath + '/' + wav2lipFolderName


sr = librosa.get_samplerate(audio_file)
print("原始sample_rate:", sr)
sr = hp.sample_rate #16000
print("hparams sample_rate:", sr)
print(f'time now: {time.time()}')
wav = audio.load_wav(audio_file, sr)
print("wav ndim:", wav.ndim)
mel = audio.melspectrogram(wav)
print(f'time now: {time.time()}')
print(wav)
print(np.shape(wav))
print(wav.shape[0]/sr) 
d = librosa.get_duration(y=wav, sr=sr, S=None, n_fft=2048, hop_length=512, center=True, filename=None)
print("y=wav, sr=sr, S=None, n_fft=2048, hop_length=512, center=True, filename=None", d)
d = librosa.get_duration(y=wav, sr=sr, S=None, n_fft=1920, hop_length=512, center=True, filename=None)
print("y=wav, sr=sr, S=None, n_fft=2048, hop_length=512, center=True, filename=None", d)


audio.save_wav(wav, '/home/ps/yulj/PycharmProjects/pythonProject/Wav2Lip-GFPGAN/inputs/1_2.wav', 16000)
audio.save_wav(wav[0:62400], '/home/ps/yulj/PycharmProjects/pythonProject/Wav2Lip-GFPGAN/inputs/1_2.wav', 16000)
audio.save_wavenet_wav(wav, '/home/ps/yulj/PycharmProjects/pythonProject/Wav2Lip-GFPGAN/inputs/1_3.wav', 16000)



import librosa


def gen_frames(waveform, frame_length=2048, hop_length=512):
    audio_frames = librosa.util.frame(waveform, frame_length=frame_length, hop_length=hop_length).T
    print(f"音频帧的frames params: frame_length={frame_length}, hop_length={hop_length}",)
    print(f"音频帧的frames：",audio_frames)
    # 打印音频帧的形状
    print(f"音频帧的形状：", audio_frames.shape)
    print(f"音频帧的frame 0：",audio_frames[0])
    print(f"音频帧的frame 0 type：",type(audio_frames[0]))
    print(f"音频帧的frame 0 data type：",type(audio_frames[0][0]))
    print(f"音频帧的0形状：", audio_frames[0].shape)
    print("---------------------------------")



    # 读取 WAV 文件
#wav_file = 'audio.wav'
#waveform, sample_rate = librosa.load(wav_file, sr=None)
waveform = wav
# 转换为音频帧
frame_length = 2048  # 音频帧的长度
hop_length = 512    # 帧之间的重叠量
hop_length = int(0.02 * sr)
frame_length = 4 * hop_length
#gen_frames(waveform, frame_length, hop_length)
gen_frames(waveform, hp.win_size, audio.get_hop_size())
gen_frames(waveform, frame_length, hop_length)
gen_frames(waveform, 1920, hop_length)
gen_frames(waveform, 1920, 512)

mel_chunks = []

fps = 30 
mel_idx_multiplier = 80. / fps
i = 0
mel_step_size = 16
while 1:
    start_idx = int(i * mel_idx_multiplier)
    if start_idx + mel_step_size > len(mel[0]):
        mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
        break
    mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
    i += 1



print(f'shape of mel: ', np.shape(mel))
print(f'len of mel 0: ', len(mel[0]))

print("Length of mel chunks: {}".format(len(mel_chunks)))
print(f'Shape of mel chunks: ', np.shape(mel_chunks))
print(f'Len of mel chunk 0: ', len(mel_chunks[0]))


if __name__ == '__main__':
    print(f'main')

    ome_path = os.environ['HOME']
    print(f'{ome_path}')
