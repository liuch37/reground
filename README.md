# reground
This is an unofficial re-implementation of paper "ReGround: Improving Textual and Spatial Grounding at No Cost" published in ECCV 2024. The code is modified on top of [GLIGEN](https://github.com/gligen/GLIGEN). Note we only test on the Generation task with Box+Text modalities. Other tasks and modalities would follow a similar procedure.

# installation
1. Create your Python environment.

2. Install your favorite PyTorch with your hardware.

3. Install libraries.
```
pip install requirements.txt
```

4. Download a checkpoint from [huggingface](https://huggingface.co/gligen/gligen-generation-text-box/blob/main/diffusion_pytorch_model.bin), rename ```diffusion_pytorch_model.bin``` to ```checkpoint_generation_text.pth```, and place it into a new folder ```./gligen_checkpoints```.

# inference
Run the below command.
```
python gligen_inference.py
```

# results
