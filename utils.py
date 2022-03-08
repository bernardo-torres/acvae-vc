import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

"""def l1_loss(y, y_hat):

    return tf.reduce_mean(tf.abs(y - y_hat))

def l2_loss(y, y_hat):

    return tf.reduce_mean(tf.square(y - y_hat))

def cross_entropy_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
"""

def normalize_mccs(mc, norm_coefs_path):
    mcep_normalization_params = np.load(norm_coefs_path)
    mcep_mean = mcep_normalization_params['mean']
    mcep_std = mcep_normalization_params['std']
    mc_norm = (mc  - mcep_mean) / mcep_std
    return mc_norm

def unnormalize_mccs(mc, norm_coefs_path):
    mcep_normalization_params = np.load(norm_coefs_path)
    mcep_mean = mcep_normalization_params['mean']
    mcep_std = mcep_normalization_params['std']
    mc_unnorm = mc * mcep_std + mcep_mean
    return mc_unnorm


def model_save(model, model_dir, model_name):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, model_name))
    
def model_load(model, model_dir, model_name):
    #model = ACVAE(nb_label=nb_label,lambda_p=lambda_p,lambda_s=lambda_s)
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
    return model

def save_figure(figure_dir, losses, epoch):        
    if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
    losses = np.array(losses)
    losses = losses.reshape(-1, 5)
    x = np.linspace(0, len(losses), len(losses))
    losses_label = ("L1", "KLD", "AC_p", "AC_s")
    plt.figure()
    plt.plot(x, losses[:,0], label=losses_label[0])
    plt.plot(x, losses[:,1], label=losses_label[1])
    plt.plot(x, losses[:,2], label=losses_label[2])
    plt.plot(x, losses[:,3], label=losses_label[3])
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
    plt.savefig(figure_dir + "/" + "epoch_{:05}".format(epoch) + ".png")
    plt.savefig(figure_dir + "/" + "result.png")
        
    plt.figure()
    plt.plot(x, losses[:,2], label=losses_label[2])
    plt.plot(x, losses[:,3], label=losses_label[3])
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
    plt.savefig(figure_dir + "/" + "epoch_{:05}_AC".format(epoch) + ".png")
    plt.savefig(figure_dir + "/" + "result_AC.png")
    
    plt.figure()
    plt.plot(x, losses[:,0], label=losses_label[0])
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
    plt.savefig(figure_dir + "/" + "epoch_{:05}_L1".format(epoch) + ".png")
    plt.savefig(figure_dir + "/" + "result_L1.png")
    
    plt.figure()
    plt.plot(x, losses[:,1], label=losses_label[1])
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
    plt.savefig(figure_dir + "/" + "epoch_{:05}_KLD".format(epoch) + ".png")
    plt.savefig(figure_dir + "/" + "result_KLD.png")

    plt.figure()
    plt.plot(x, losses[:,4], label='Total')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
    plt.savefig(figure_dir + "/" + "epoch_{:05}_total".format(epoch) + ".png")
    plt.savefig(figure_dir + "/" + "result_total.png")