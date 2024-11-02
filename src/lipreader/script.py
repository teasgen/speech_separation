import numpy as np
import torch
from src.lipreader.lipreading.utils import load_model, save2npz, load_json
from src.lipreader.lipreading.model import Lipreading
from src.lipreader.lipreading.dataloaders import get_preprocessing_pipelines

def extract_feats(model, mouth_patch_path):
    model.eval()
    preprocessing_func = get_preprocessing_pipelines(modality="video")['test']
    data = preprocessing_func(np.load(mouth_patch_path)['data'])  # data: TxHxW
    return model(torch.FloatTensor(data)[None, None, :, :, :].to(torch.device("cpu")), lengths=[data.shape[0]])

def load_model_from_json(config_path, model_path, modality='video', num_classes=500):
    args_loaded = load_json(config_path)
    
    tcn_options = {
        'num_layers': args_loaded.get('tcn_num_layers', 4),
        'kernel_size': args_loaded.get('tcn_kernel_size', [3]),
        'dropout': args_loaded.get('tcn_dropout', 0.2),
        'dwpw': args_loaded.get('tcn_dwpw', False),
        'width_mult': args_loaded.get('tcn_width_mult', 1),
    }
    
    model = Lipreading(
        modality=modality,
        num_classes=num_classes,
        backbone_type=args_loaded['backbone_type'],
        width_mult=args_loaded['width_mult'],
        relu_type=args_loaded['relu_type'],
        tcn_options=tcn_options,
        use_boundary=args_loaded.get("use_boundary", False),
        extract_feats=True
    ).to(torch.device("cpu"))
    
    model = load_model(model_path, model, allow_size_mismatch=True)
    return model

if __name__ == '__main__':
    config_path = "configs/lrw_snv1x_tcn1x.json"
    model_path = "lrw_snv1x_tcn1x.pth"
    mouth_patch_path = "data/mouth_patches/00000002026.npz"
    output_path = "embeddings/sample_1.npz"

    model = load_model_from_json(config_path, model_path)
    embeddings = extract_feats(model, mouth_patch_path)
    save2npz(output_path, data=embeddings.cpu().detach().numpy())
