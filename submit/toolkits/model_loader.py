import torch
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from spiga.demo.visualize.plotter import Plotter

device = 'cpu'

def load_spiga_framework(dataset='wflw', device='cpu'):

    face_processor = SPIGAFramework(ModelConfig(dataset))
    return face_processor



if __name__ == '__main__':
    print(torch.cuda.is_available())
    processor = load_spiga_framework()
    print(processor.model)