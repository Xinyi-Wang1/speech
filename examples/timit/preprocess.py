from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import glob
import json
import os
import random
import tqdm
import sys
from random import randint

from speech.utils import data_helpers
from speech.utils import wave

WAV_EXT = "WAV"

TEST_SPEAKERS = ['MDAB0', 'MWBT0', 'FELC0', 'MTAS1', 'MWEW0', \
    'FPAS0', 'MJMP0', 'MLNT0', 'FPKT0', 'MLLL0', 'MTLS0', 'FJLM0', \
    'MBPM0', 'MKLT0', 'FNLP0', 'MCMJ0', 'MJDH0', 'FMGD0', 'MGRT0', \
    'MNJM0', 'FDHC0', 'MJLN0', 'MPAM0', 'FMLD0']

P = 0.9

def load_phone_map():
    with open("/Users/xinyiwang/Documents/GitHub/speech/examples/timit/phones.60-48-39.map", 'r') as fid:
        lines = (l.strip().split() for l in fid)
        lines = [l for l in lines if len(l) == 3]
    m60_48 = {l[0] : l[1] for l in lines}
    m48_39 = {l[1] : l[2] for l in lines}
    return m60_48, m48_39

def load_transcripts(path):
    pattern = os.path.join(path, "*/*/*.PHN")
    print(pattern)
    m60_48, _ = load_phone_map()
    files = glob.glob(pattern)
    # Standard practic is to remove all "sa" sentences
    # for each speaker since they are the same for all.
    filt_sa = lambda x : os.path.basename(x)[:2] != "sa"
    files = filter(filt_sa, files)
    data = {}
    z = {}
    for f in files:
        with open(f) as fid:
            lines = (l.strip() for l in fid)
            phonemes = (l.split()[-1] for l in lines)
            phonemes = [m60_48[p] for p in phonemes if p in m60_48]
            data[f] = phonemes
        chance = random.random() < P
        if not chance:
            z[f] = ''
        else:
            length = len(phonemes)
            index = randint(1,length-1)
            z[f] = phonemes[index]
    print("size of data: ", sys.getsizeof(data))
    return data, z

def split_by_speaker(data, dev_speakers=50):

    def speaker_id(f):
        return os.path.basename(os.path.dirname(f))
    speaker_dict = collections.defaultdict(list)
    for k, v in data.items():
        speaker_dict[speaker_id(k)].append((k, v))
    speakers = list(speaker_dict)
    for t in TEST_SPEAKERS:
        speakers.remove(t)
    random.shuffle(speakers)
    dev = speakers[:dev_speakers]
    dev = dict(v for s in dev for v in speaker_dict[s])
    test = dict(v for s in TEST_SPEAKERS for v in speaker_dict[s])
    return dev, test

def convert_to_wav(path):
    data_helpers.convert_full_set(path, "*/*/*/*.wav",
            new_ext=WAV_EXT,
            use_avconv=False)

def build_json(data, path, set_name):
    basename = set_name + os.path.extsep + "json"
    with open(os.path.join(path, basename), 'w') as fid:
        for k, t in tqdm.tqdm(data.items()):
            wave_file = os.path.splitext(k)[0] + os.path.extsep + WAV_EXT
            dur = wave.wav_duration(wave_file)
            datum = {'text' : t,
                     'duration' : dur,
                     'audio' : wave_file}
            json.dump(datum, fid)
            fid.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Preprocess Timit dataset.")

    parser.add_argument("output_directory",
        help="Path where the dataset is saved.")
    args = parser.parse_args()

    path = os.path.join(args.output_directory, "raw")
#    print("\n")
#    print(path)
    path = os.path.abspath(path)

    print("Converting files from NIST to standard wave format...")
    convert_to_wav(path)

    print("Preprocessing train")
    train, contextTrain = load_transcripts(os.path.join(path, "TRAIN"))
#    print(path)
    build_json(train, path, "train")
    build_json(contextTrain, path, "contextTrain")

    print("Preprocessing dev")
    transcripts, contextTest = load_transcripts(os.path.join(path, "TEST"))
    dev, test = split_by_speaker(transcripts)
    build_json(dev, path, "dev")

    print("Preprocessing test")
    build_json(test, path, "test")

    build_json(contextTest, path, "contextTest")
