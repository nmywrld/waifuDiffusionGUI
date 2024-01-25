from transformers.tools.base import Tool, get_default_device
from transformers.utils import is_accelerate_available
import torch

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


TEXT_TO_IMAGE_DESCRIPTION = (
    "This is a tool that creates an image according to a prompt, which is a text description. It takes an input named `prompt` which "
    "contains the image description and outputs an image."
)


class TextToImageTool(Tool):
    default_checkpoint = "runwayml/stable-diffusion-v1-5"
    description = TEXT_TO_IMAGE_DESCRIPTION
    inputs = ['text']
    outputs = ['image']

    def __init__(self, device=None, **hub_kwargs) -> None:
        if not is_accelerate_available():
            raise ImportError("Accelerate should be installed in order to use tools.")

        super().__init__()

        self.device = device
        self.pipeline = None
        self.hub_kwargs = hub_kwargs

    def setup(self):
        if self.device is None:
            self.device = get_default_device()

        self.pipeline = DiffusionPipeline.from_pretrained(self.default_checkpoint)
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.to(self.device)

        if self.device.type == "cuda":
            self.pipeline.to(torch_dtype=torch.float16)

        self.is_initialized = True

    def __call__(self, prompt):
        if not self.is_initialized:
            self.setup()

        negative_prompt = "low quality, bad quality, deformed, low resolution"
        added_prompt = " , highest quality, highly realistic, very high resolution"

        return self.pipeline(prompt + added_prompt, negative_prompt=negative_prompt, num_inference_steps=25).images[0]

