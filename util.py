import soundfile
import librosa
import numpy as np

def read_audio(path, target_fs=None):
    """ Read a wav file
    Args:
        path: str, path of the audio file
        target_fs: int, (Default value = None) sampling rate of the returned audio file, if not specified, the sampling
            rate of the audio file is taken

    Returns:
        tuple
        (numpy.array, sampling rate), array containing the audio at the sampling rate given

    """
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs