import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import pyworld
import librosa
import time
import matplotlib.pyplot as plt

from preprocess import *
from model import ACVAE
from utils import model_load, normalize_mccs, unnormalize_mccs

import soundfile as sf

ALL = 0
SINGLE_FILE = 1
mode = ALL
voice_dir_list = ["SF1", "SF2", "SM1", "SM2"]

parser = argparse.ArgumentParser(description='ACVAE-VC Testing script')
parser.add_argument('--data', type=str, default='./data/vcc2016_training/', metavar='D',
                    help="folder where data is located")
parser.add_argument('--file', type=str, default='default', metavar='F',
                    help="file to be converted")
parser.add_argument('--n_speakers', type=int, default=4, metavar='N',
                    help='number of target speakers. size of label c')
parser.add_argument('--source', type=int, default=0, metavar='S',
                    help='Source speaker in range [0, n_speakers-1')
parser.add_argument('--target', type=int, default=0, metavar='T',
                    help='target speaker in range [0, n_speakers-1')
parser.add_argument('--model-dir', type=str, default='/model/', metavar='D',
                    help="folder where model is located")
parser.add_argument('--model', type=str, default='', metavar='MOD',
                    help='model name to choose from.')
parser.add_argument('--output-dir', type=str, default='./output/', metavar='O',
                    help='output directory')
parser.add_argument('--sampling-rate', type=int, default=16000, metavar='SR',
                    help='sampling rate')
parser.add_argument('--frame-period', type=float, default=5.0, metavar='FP',
                    help='frame period (ms)')
args = parser.parse_args()


if args.file is not 'default':
  mode = SINGLE_FILE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ACVAE(nb_label=args.n_speakers,lambda_p=0,lambda_s=0).to(device)
model = model_load(model, args.model_dir, args.model)


def conv(voice_file, s_label, t_label):
  print("Conversion Start.")
  print("Source File:" + voice_file)

  if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

  voice_path_s = os.path.join(args.data, voice_dir_list[s_label])
  voice_path_t = os.path.join(args.data, voice_dir_list[t_label])

  wav, _ = librosa.load(os.path.join(voice_path_s, voice_file), sr = args.sampling_rate, mono = True)
  wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
  wav = wav_padding(wav = wav, sr = args.sampling_rate, frame_period = args.frame_period, multiple = 4)
  f0, timeaxis, sp, ap, mc = world_decompose(wav = wav, fs = args.sampling_rate, frame_period = args.frame_period)

  mc_transposed  = np.array(mc).T

  
  norm_coefs_path_s = os.path.join(voice_path_s, "mcep_"+voice_dir_list[s_label]+".npz")
  norm_coefs_path_t = os.path.join(voice_path_t, "mcep_"+voice_dir_list[t_label]+".npz")
  mc_norm = normalize_mccs(mc_transposed, norm_coefs_path_s)

  x = torch.Tensor(mc_norm).view(1, 1, mc_norm.shape[0], mc_norm.shape[1])

  label_s_tensor = torch.Tensor(np.array([s_label])).view(1, 1)
  label_t_tensor = torch.Tensor(np.array([t_label])).view(1, 1)

  x = x.to(device)
  label_s_tensor = label_s_tensor.to(device)
  label_t_tensor = label_t_tensor.to(device)

  # Extracts latent state and uses it to reconstruct target
  mu_enc, logvar_enc = model.encode(x, label_s_tensor)
  z_enc = model.reparameterize(mu_enc, logvar_enc)
  # x^
  mu_dec_t, logvar_dec_t = model.decode(z_enc, label_t_tensor)
  z_dec_t = model.reparameterize(mu_dec_t, logvar_dec_t)
  
  if (torch.cuda.is_available()):
      z_dec_t = z_dec_t.data.cpu().numpy().reshape((mc_norm.shape[0], mc_norm.shape[1]))
  else:
      z_dec_t = z_dec_t.data.numpy().reshape((mc_norm.shape[0], mc_norm.shape[1]))
  # x_
  # Using same latent state use it to reconstruct source
  mu_dec_s, logvar_dec_s = model.decode(z_enc, label_s_tensor)
  z_dec_s = model.reparameterize(mu_dec_s, logvar_dec_s)

  if (torch.cuda.is_available()):
      z_dec_s = z_dec_s.data.cpu().numpy().reshape((mc_norm.shape[0], mc_norm.shape[1]))
  else:
      z_dec_s = z_dec_s.data.numpy().reshape((mc_norm.shape[0], mc_norm.shape[1]))

  #mc_converted_t = z_dec_t * mcep_std_t + mcep_mean_t
  mc_converted_t = unnormalize_mccs(z_dec_t, norm_coefs_path_t)
  mc_converted_t = mc_converted_t.T
  mc_converted_t = np.ascontiguousarray(mc_converted_t)
  sp_converted_t = world_decode_mc(mc = mc_converted_t, fs = args.sampling_rate)
  #mc_converted_s = z_dec_s * mcep_std_s + 
  mc_converted_s = unnormalize_mccs(z_dec_s, norm_coefs_path_s)
  mc_converted_s = mc_converted_s.T
  mc_converted_s = np.ascontiguousarray(mc_converted_s)
  sp_converted_s = world_decode_mc(mc = mc_converted_s, fs = args.sampling_rate)

  sp_gained = np.multiply(sp, np.divide(sp_converted_t, sp_converted_s))

  logf0s_normalization_params_s = np.load(os.path.join(voice_path_s, "log_f0_"+voice_dir_list[s_label]+".npz"))
  logf0s_mean_s = logf0s_normalization_params_s['mean']
  logf0s_std_s = logf0s_normalization_params_s['std']
  logf0s_normalization_params_t = np.load(os.path.join(voice_path_t, "log_f0_"+voice_dir_list[t_label]+".npz"))
  logf0s_mean_t = logf0s_normalization_params_t['mean']
  logf0s_std_t = logf0s_normalization_params_t['std']


  f0_converted = pitch_conversion(f0 = f0, mean_log_src = logf0s_mean_s, std_log_src = logf0s_std_s, mean_log_target = logf0s_mean_t, std_log_target = logf0s_std_t)
  
  wav_transformed = world_speech_synthesis(f0 = f0_converted, sp = sp_gained, ap = ap, fs = args.sampling_rate, frame_period = args.frame_period)
  sf.write(os.path.join(args.output_dir, voice_dir_list[s_label]+"_to_"+voice_dir_list[t_label]+"_["+args.file+"].wav"),
            wav_transformed, 
            args.sampling_rate)
  wav_source = world_speech_synthesis(f0 = f0_converted, sp = sp, ap = ap, fs = args.sampling_rate, frame_period = args.frame_period)
  sf.write(os.path.join(args.output_dir, voice_dir_list[s_label]+"_to_"+voice_dir_list[t_label]+"_["+args.file+"]_nonconv.wav"),
            wav_source, 
            args.sampling_rate)
  print("Converted: " + voice_dir_list[s_label] + " -> " + voice_dir_list[t_label])

if mode == SINGLE_FILE:
  conv(args.file, args.source, args.target)
