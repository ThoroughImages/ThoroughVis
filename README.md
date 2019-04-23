[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align=center><img alt="ThoroughVis" width="400" src="https://raw.githubusercontent.com/ThoroughImages/ThoroughVis/master/figs/ThoroughVis.png"/></div>

#### East to Use

```bash
python conv_vis.py            
   --model                    # Checkpoint path.
   --image_path               # Path of the input image for CNN.
   --output_dir               # Output directory for feature maps. 
```

The model repo should contain (for example):

```bash
export.ckpt:      Trained model parameters.
export.ckpt.meta: Model structure.
```
#### Search for Data Entrance Automatically

Given the input image, the program automatically finds all the `Placeholders` in the computing graph and searches for the best-matched `ones` to feed the image into.

Two debug trials once ThoroughVis fails:
1. Resize the input image to match the target data entrance.
2. Make sure the target `Placeholder` is correctly defined in the computing graph.

#### Default Placeholder Feed-In

The program will automatically acquire all the `Placeholders` and feed them with default zero values to make the computing graph flow properly.

```python
tf.bool:         False
tf.int32:        0
tf.int64:        0
tf.float16:      0.0
tf.float32:      0.0
tf.array(shape): numpy.zeros(shape)
```

Our team will add self-defined feed-in support in the next update.

#### Minimum Requirement
```bash
tensorflow
numpy 
matplotlib
uuid
```

#### Licence
MIT
