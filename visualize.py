import argparse

import torch
from torchview import draw_graph

from models.common import DetectMultiBackend


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLOv5 model architecture")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for visualization")
    args = parser.parse_args()

    # Load YOLOv5 model
    device = 0 if torch.cuda.is_available() else "cpu"
    model = DetectMultiBackend(args.weights, device=torch.device(device))

    # Visualize model graph
    input_size = (args.batch_size, 3, args.imgsz, args.imgsz)
    model_graph = draw_graph(model, input_size=input_size, device="meta", expand_nested=True)
    model_graph.visual_graph.render(filename="model_graph", format="png")
    print("Model graph saved as model_graph.png")


if __name__ == "__main__":
    main()
