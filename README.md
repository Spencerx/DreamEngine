



<h1 align="center"> DreamEngine</h1>

<p align="center">



<a href="https://arxiv.org/abs/2502.20172">
<img alt="Static Badge" src="https://img.shields.io/badge/Paper-arXiv-red">
</a>

<a href="https://huggingface.co/leonardPKU/DreamEngine-ObjectFusion">
<img alt="Static Badge" src="https://img.shields.io/badge/Model-HuggingFace-yellow">
</a>


</p>

<img width="898" alt="截屏2025-02-23 22 38 04" src="https://github.com/user-attachments/assets/cdeb00d3-2c6a-459a-b884-7946a51075b8" />

DreamEngine is a unified framework that integrates multimodal encoders like QwenVL with diffusion models through a two-stage training approach, enabling advanced text-image interleaved control and achieving state-of-the-art performance in generating images with complex, concept-merged inputs. 


https://github.com/user-attachments/assets/f070c0e8-4ea2-4294-b878-63f564da1b79




Updates:
- 2025-03-03: Release checkpoint and a demo for text-guided object fusion.


## Run the Demo locally

```bash
bash setup.sh

# setup the paths in demo.py
python src/scripts/eval/demo.py

```


## Model Structure

<img width="1136" alt="截屏2025-02-27 23 14 47" src="https://github.com/user-attachments/assets/ce5bf658-e571-440a-ad10-6b4e68fb6c4a" />


## Training

<img width="1136" alt="截屏2025-02-27 23 15 16" src="https://github.com/user-attachments/assets/bb47b53b-e512-43b1-a791-ee066ee52927" />


## Demos


<img width="1136" alt="截屏2025-02-27 23 15 03" src="https://github.com/user-attachments/assets/025af313-9360-4602-a4cc-78c7a4fb7137" />
<img width="1136" alt="截屏2025-02-27 23 15 24" src="https://github.com/user-attachments/assets/94fcf73f-3f1d-4baa-8a81-edb9d3fc3c42" />
<img width="1136" alt="截屏2025-02-27 23 15 30" src="https://github.com/user-attachments/assets/3e624f79-7d24-4b14-aa9d-46a8eee0c271" />

## Citation

If you feel the work helpful, please kindly cite

```bibtex
@misc{chen2025multimodalrepresentationalignmentimage,
      title={Multimodal Representation Alignment for Image Generation: Text-Image Interleaved Control Is Easier Than You Think}, 
      author={Liang Chen and Shuai Bai and Wenhao Chai and Weichu Xie and Haozhe Zhao and Leon Vinci and Junyang Lin and Baobao Chang},
      year={2025},
      eprint={2502.20172},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.20172}, 
}
```
