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

# reground modification
Theoretically from the paper, the following lines in ```ldm/modules/attention.py```
```
    def forward(self, x, context, objs):
#        return checkpoint(self._forward, (x, context, objs), self.parameters(), self.use_checkpoint)
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, context, objs)
        else:
            return self._forward(x, context, objs)
```
can be replaced with
```
    def forward(self, x, context, objs):
#        return checkpoint(self._forward_reground, (x, context, objs), self.parameters(), self.use_checkpoint)
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, context, objs)
        else:
            return self._forward_reground(x, context, objs)
```

# inference
Run the below command.
```
python gligen_inference.py
```

# results
Prompt = "a teddy bear blowing smoke sitting next to a bird".

## GLIGEN
<p float="left">
  <img src="https://github.com/liuch37/reground/blob/master/results/gligen_1.png" width="20%" />
  <img src="https://github.com/liuch37/reground/blob/master/results/gligen_2.png" width="20%" />
  <img src="https://github.com/liuch37/reground/blob/master/results/gligen_3.png" width="20%" />
  <img src="https://github.com/liuch37/reground/blob/master/results/gligen_4.png" width="20%" />
  <img src="https://github.com/liuch37/reground/blob/master/results/gligen_5.png" width="20%" />
</p>

## REGROUND
Can not reproduce.