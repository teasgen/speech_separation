import time
import torch
import torch.nn as nn
 
from src.utils.init_utils import init_lipreader
from src.model import DPTNAVWavEncDec
 
device = "cuda:7"
 
class VideoSSModel(nn.Module):
    def __init__(self, ss: nn.Module):
        super().__init__()
        self.lipreader = init_lipreader("src/lipreader/configs/lrw_resnet18_mstcn.json", "lrw_resnet18_mstcn_video.pth").to(device)
        self.ss = ss
        self.lipreader.eval()
 
    def forward(self, **batch):
        video = batch["video"]
        embed = self.lipreader(video, lengths=[50]).permute(0, 2, 1)
        batch["s1_embedding"] = embed[:1, ...]
        batch["s2_embedding"] = embed[1:2, ...]
        self.ss(**batch)
 
 
def run_profiler(model, **batch):
    model(**batch)
    inference_time = []
    with torch.no_grad(), torch.profiler.profile(with_flops=True, profile_memory=True) as prof:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        model(**batch)
 
    for _ in range(10):
        start_time = time.time()
        model(**batch)
        inference_time.append(time.time() - start_time)
    
    events = prof.key_averages()
 
    total_flops = 0
    total_memory = 0
    for event in events:
        if event.flops:
            total_flops += event.flops
        if hasattr(event, 'cuda_memory_usage'):
            total_memory += event.cuda_memory_usage
 
    gflops = total_flops / 1e9
    memory_mb = total_memory / 1e6
 
    results = {
        "gflops": gflops,
        "cuda_memory": memory_mb,
        "inference_time_mean": torch.tensor(inference_time).mean().item(),
        "inference_time_std": torch.tensor(inference_time).std().item(),
        "num_params": sum([p.numel() for p in model.parameters() if p.requires_grad])
    }
 
    return results
 
bs = 1
 
mix = torch.randn((bs, 32000)).to(device)
video = torch.randn((2, 1, 50, 88, 88)).to(device)
 
batch = {
    "video": video,
    "mix": mix,
}
 
ss_model = DPTNAVWavEncDec(
    video_emb_size = 512,
    num_features = 128,
    hidden_video = 128,
    kernel_size_enc = 7,
    hidden_dim = 128,
    num_blocks = 6,
    chunk_size = 150,
    step_size = 75,
    dropout = 0.1,
    num_heads = 4,
    bidir = True,
).to(device)
 
av_model = VideoSSModel(ss_model)
 
run_profiler(av_model, **batch)