# -*- coding: utf-8 -*-

import os
import librosa
import numpy as np
from hparam import hparam as hp
import utils

# downloaded dataset path
audio_path = hp.unprocessed_data

def save_spectrogram_tisv():
    
    print("start text independent utterance feature extraction")
    os.makedirs(hp.data.train_path, exist_ok=True)   
    os.makedirs(hp.data.test_path, exist_ok=True)    

    utter_min_len = (
        hp.data.tisv_frame * hp.data.hop + hp.data.window
    ) * hp.data.sr  # lower bound of utterance length
    total_speaker_num = len(os.listdir(audio_path))
    
    train_speaker_num = total_speaker_num
    
    for i, folder in enumerate(os.listdir(audio_path)):
        print("%dth speaker %s processing..." % (i+1, folder))
        utterances_spec = []
        dir_path = os.path.join(audio_path, folder)
        for utter_name in os.listdir(dir_path)[:10]:
            if utter_name[-4:].upper() == '.WAV':
                utter_path = os.path.join(dir_path, utter_name)         
                utter, sr = librosa.core.load(utter_path, hp.data.sr)       

                utter = utils.vad(utter)
                
                S = librosa.core.stft(y=utter, n_fft=hp.data.nfft,
                                      win_length=int(hp.data.window * sr),
                                      hop_length=int(hp.data.hop * sr))
                S = np.abs(S) ** 2
                mel_basis = librosa.filters.mel(
                    sr=hp.data.sr, n_fft=hp.data.nfft, 
                    n_mels=hp.data.nmels)
                S = np.log10(np.dot(mel_basis, S) + 1e-6)
                S = S.T
                for idx in range(0, len((S))-hp.data.tisv_frame, 
                                 hp.data.tisv_frame):
                    utterances_spec.append(
                        S[idx:idx+hp.data.tisv_frame].T
                    )

        utterances_spec = np.array(utterances_spec)
        if i < train_speaker_num:      # save spectrogram as numpy file
            np.save(
                os.path.join(
                    hp.data.train_path, "%s.npy" % folder
                ), 
                utterances_spec
            )
        else:
            np.save(
                os.path.join(
                    hp.data.test_path, "%s.npy" % folder
                ), 
                utterances_spec
            )


if __name__ == "__main__":
    save_spectrogram_tisv()
