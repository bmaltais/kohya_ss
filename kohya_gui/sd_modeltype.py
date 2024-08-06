from os.path import isfile
from safetensors import safe_open
import enum

# methodology is based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/82a973c04367123ae98bd9abdf80d9eda9b910e2/modules/sd_models.py#L379-L403

class ModelType(enum.Enum):
    UNKNOWN = 0
    SD1 = 1
    SD2 = 2
    SDXL = 3
    SD3 = 4

class SDModelType:
    def __init__(self, safetensors_path):
        self.model_type = ModelType.UNKNOWN

        if not isfile(safetensors_path):
            return

        try:
            st = safe_open(filename=safetensors_path, framework="numpy", device="cpu")
            def hasKeyPrefix(pfx):
                return any(k.startswith(pfx) for k in st.keys())

            if "model.diffusion_model.x_embedder.proj.weight" in st.keys():
                self.model_type = ModelType.SD3
            elif hasKeyPrefix("conditioner."):
                self.model_type = ModelType.SDXL
            elif hasKeyPrefix("cond_stage_model.model."):
                self.model_type = ModelType.SD2
            elif hasKeyPrefix("model."):
                self.model_type = ModelType.SD1
        except:
            pass

    def Is_SD1(self):
        return self.model_type == ModelType.SD1

    def Is_SD2(self):
        return self.model_type == ModelType.SD2

    def Is_SDXL(self):
        return self.model_type == ModelType.SDXL

    def Is_SD3(self):
        return self.model_type == ModelType.SD3
