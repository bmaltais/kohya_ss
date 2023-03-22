import torch

def apply_snr_weight(loss, noisy_latents, latents, gamma):
    gamma = gamma
    if gamma:
        sigma = torch.sub(noisy_latents, latents)
        zeros = torch.zeros_like(sigma) 
        alpha_mean_sq = torch.nn.functional.mse_loss(latents.float(), zeros.float(), reduction="none").mean([1, 2, 3])
        sigma_mean_sq = torch.nn.functional.mse_loss(sigma.float(), zeros.float(), reduction="none").mean([1, 2, 3])
        snr = torch.div(alpha_mean_sq, sigma_mean_sq)
        gamma_over_snr = torch.div(torch.ones_like(snr) * gamma, snr)
        snr_weight = torch.minimum(gamma_over_snr, torch.ones_like(gamma_over_snr)).float()
        loss = loss * snr_weight
    return loss
