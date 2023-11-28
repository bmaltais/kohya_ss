import threading
import torch
from torchvision import transforms
from typing import *
from diffusers import EulerAncestralDiscreteScheduler
import diffusers.schedulers.scheduling_euler_ancestral_discrete
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteSchedulerOutput


def fire_in_thread(f, *args, **kwargs):
    threading.Thread(target=f, args=args, kwargs=kwargs).start()


# TODO make inf_utils.py


# region Gradual Latent hires fix


class GradualLatent:
    def __init__(
        self,
        ratio,
        start_timesteps,
        every_n_steps,
        ratio_step,
        s_noise=1.0,
        gaussian_blur_ksize=None,
        gaussian_blur_sigma=0.5,
        gaussian_blur_strength=0.5,
    ):
        self.ratio = ratio
        self.start_timesteps = start_timesteps
        self.every_n_steps = every_n_steps
        self.ratio_step = ratio_step
        self.s_noise = s_noise
        self.gaussian_blur_ksize = gaussian_blur_ksize
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.gaussian_blur_strength = gaussian_blur_strength

    def __str__(self) -> str:
        return (
            f"GradualLatent(ratio={self.ratio}, start_timesteps={self.start_timesteps}, "
            + f"every_n_steps={self.every_n_steps}, ratio_step={self.ratio_step}, s_noise={self.s_noise}, "
            + f"gaussian_blur_ksize={self.gaussian_blur_ksize}, gaussian_blur_sigma={self.gaussian_blur_sigma}, gaussian_blur_strength={self.gaussian_blur_strength})"
        )

    def apply_unshark_mask(self, x: torch.Tensor):
        if self.gaussian_blur_ksize is None:
            return x
        blurred = transforms.functional.gaussian_blur(x, self.gaussian_blur_ksize, self.gaussian_blur_sigma)
        # mask = torch.sigmoid((x - blurred) * self.gaussian_blur_strength)
        mask = (x - blurred) * self.gaussian_blur_strength
        sharpened = x + mask
        return sharpened


class EulerAncestralDiscreteSchedulerGL(EulerAncestralDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resized_size = None
        self.gradual_latent = None

    def set_gradual_latent_params(self, size, gradual_latent: GradualLatent):
        self.resized_size = size
        self.gradual_latent = gradual_latent

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """

        if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor) or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            # logger.warning(
            print(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`")

        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt

        device = model_output.device
        if self.resized_size is None:
            noise = diffusers.schedulers.scheduling_euler_ancestral_discrete.randn_tensor(
                model_output.shape, dtype=model_output.dtype, device=device, generator=generator
            )
            s_noise = 1.0
        else:
            print(
                "resized_size", self.resized_size, "model_output.shape", model_output.shape, "prev_sample.shape", prev_sample.shape
            )
            org_dtype = prev_sample.dtype
            if org_dtype == torch.bfloat16:
                prev_sample = prev_sample.float()

            prev_sample = torch.nn.functional.interpolate(
                prev_sample.float(), size=self.resized_size, mode="bicubic", align_corners=False
            ).to(dtype=org_dtype)

            # apply unsharp mask / アンシャープマスクを適用する
            if self.gradual_latent.gaussian_blur_ksize:
                prev_sample = self.gradual_latent.apply_unshark_mask(prev_sample)

            noise = diffusers.schedulers.scheduling_euler_ancestral_discrete.randn_tensor(
                (model_output.shape[0], model_output.shape[1], self.resized_size[0], self.resized_size[1]),
                dtype=model_output.dtype,
                device=device,
                generator=generator,
            )
            s_noise = self.gradual_latent.s_noise

        prev_sample = prev_sample + noise * sigma_up * s_noise

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return EulerAncestralDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


# endregion
