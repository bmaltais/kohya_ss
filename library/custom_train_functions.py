import torch
import argparse

def apply_snr_weight(loss, latents, noisy_latents, gamma):
  sigma = torch.sub(noisy_latents, latents) #find noise as applied by scheduler
  zeros = torch.zeros_like(sigma) 
  alpha_mean_sq = torch.nn.functional.mse_loss(latents.float(), zeros.float(), reduction="none").mean([1, 2, 3]) #trick to get Mean Square/Second Moment
  sigma_mean_sq = torch.nn.functional.mse_loss(sigma.float(), zeros.float(), reduction="none").mean([1, 2, 3]) #trick to get Mean Square/Second Moment
  snr = torch.div(alpha_mean_sq,sigma_mean_sq) #Signal to Noise Ratio = ratio of Mean Squares
  gamma_over_snr = torch.div(torch.ones_like(snr)*gamma,snr)
  snr_weight = torch.minimum(gamma_over_snr,torch.ones_like(gamma_over_snr)).float() #from paper
  loss = loss * snr_weight
  #print(snr_weight)
  return loss

def add_custom_train_arguments(parser: argparse.ArgumentParser):
  parser.add_argument("--min_snr_gamma", type=float, default=0, help="gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by paper.")
