import argparse
import numpy as np
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from PIL import Image
from visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(description='Visualization for CNN.')
    parser.add_argument('--model', help='Checkpoint path.', type=str)
    parser.add_argument('--image_path', help='Path of the input image.', type=str)
    parser.add_argument('--output_dir', help='Path of the output images.', type=str, default='./outputs')
    args = parser.parse_args()

    # Read the image as input.
    image_input = np.array(Image.open(args.image_path))

    # Visualize the feature maps for CNN model.
    vis = Visualizer(meta_graph=args.model, output_dir=args.output_dir)
    vis.plot_conv_outputs(image_input)


if __name__ == '__main__':
    main()
