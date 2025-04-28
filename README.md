# Human Motion Transfer using Zero-Shot Image Diffusion

## Overview

This project explores **zero-shot human motion transfer** using a combination of:
- **Image Diffusion Models**
- **ControlNet**
- **IP-Adapter**
- The method described in the paper **[RAVE](https://arxiv.org/abs/2402.14653)** (Zero-Shot Video Editing with Pretrained Diffusion Models).

The goal is to achieve realistic and temporally consistent human motion transfer without the need for additional fine-tuning or training. By building upon recent advancements in diffusion-based video editing, the project preserves both the motion and semantic structure of input content in a highly efficient and scalable manner.

## Key Components

- **ControlNet**: Guides the generation process by conditioning on structural information (e.g., poses, edges, depth maps).
- **IP-Adapter**: Provides flexible, plug-and-play control over appearance features in diffusion models.
- **RAVE (Noise Shuffling Strategy)**: 
  - A zero-shot video editing approach using pre-trained text-to-image diffusion models.
  - Preserves spatio-temporal consistency by shuffling noise across frames, leading to coherent outputs.
  - Faster and more memory-efficient than traditional frame-by-frame editing methods.

## How it Works

1. **Input**: A reference image (source appearance) and a target video (motion).
2. **Feature Extraction**:
   - Pose or motion control maps are extracted from the video frames.
3. **Conditioning**:
   - ControlNet is used to condition the diffusion process on the extracted poses.
   - IP-Adapter injects the appearance information from the reference image.
4. **Diffusion-based Generation**:
   - A pre-trained text-to-image diffusion model is leveraged.
   - RAVE's noise shuffling strategy ensures temporal coherence across video frames without requiring retraining.
5. **Output**: A new video where the human subject adopts the motion from the target video while preserving the appearance of the reference image.

## Advantages

- **Zero-Shot**: No retraining or fine-tuning needed on custom datasets.
- **High Quality**: Realistic transfer of appearance and motion with minimal artifacts.
- **Temporal Consistency**: Thanks to the noise shuffling technique from RAVE.
- **Memory Efficiency**: Suitable for processing longer video sequences without high resource demands.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/human-motion-transfer-diffusion.git
   cd human-motion-transfer-diffusion
   pip install -r requirements.txt
   ```

2. Prepare inputs reference image (main content) and target video (motion input), then change the config file in the configs directory 

3. Run the pipeline:
   ```bash
   python run_experiment.py configs/IP-controlnet.yaml
   ```

## References

- [RAVE: Zero-Shot Video Editing with Pretrained Diffusion Models](https://arxiv.org/abs/2402.14653)
- [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)
- [IP-Adapter: Text-to-Image Diffusion Models are Zero-Shot Controllers](https://arxiv.org/abs/2308.06721)

## Future Work

- Explore background and lighting transfer alongside motion.
- Fine-grained control over transfer strength (partial vs full motion adaptation).

## Acknowledgments

Thanks to the developers and researchers behind ControlNet, IP-Adapter, and RAVE for making their models and methods publicly available.
```
