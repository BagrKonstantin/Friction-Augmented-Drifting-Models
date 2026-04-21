import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy import linalg

# Metrics imports
from OFM.ofmsrc.inception import InceptionV3
from OFM.ofmsrc.fid_score import calculate_frechet_distance
from OFM.ALAE.alae_ffhq_inference import load_model, decode

class MetricsEvaluator:
    def __init__(self, device, compute_metrics=True):
        self.device = device
        self.compute_metrics = compute_metrics
        self.alae_model = None
        self.inception_model = None
        self.clip_model = None
        
        if self.compute_metrics:
            self._setup_models()

    def _setup_models(self):
        print("Setting up Evaluation models (FID & CMMD)...")
        self.alae_model = load_model("../OFM/ALAE/configs/ffhq.yaml",
                                    training_artifacts_dir="../OFM/ALAE/training_artifacts/ffhq/").to(self.device)
        self.inception_model = InceptionV3([3]).to(self.device)
        self.inception_model.eval()

        # CMMD setup
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmmd_src = os.path.join(root_dir, "cmmd_pytorch", "src")
        if cmmd_src not in sys.path:
            sys.path.append(cmmd_src)
        
        try:
            from cmmd_pytorch.embedding import ClipEmbeddingModel
            from cmmd_pytorch.distance import mmd
            self.clip_model = ClipEmbeddingModel()
            self.mmd_func = mmd
        except ImportError:
            print("Warning: cmmd_pytorch not found. CMMD evaluation will be disabled.")
            self.clip_model = None

    def get_eval_features(self, latents, batch_size=32):
        if not self.compute_metrics:
            return None, None
            
        fid_act = []
        cmmd_act = []
        with torch.no_grad():
            for i in tqdm(range(0, len(latents), batch_size), desc="Extracting features"):
                batch = latents[i:i+batch_size].to(self.device)

                # ALAE Decode (to [-1, 1])
                imgs = decode(self.alae_model, batch)
                # Normalize to [0, 1]
                imgs_01 = (imgs * 0.5 + 0.5).clamp(0, 1)

                # --- FID features ---
                feat_fid = self.inception_model(imgs_01)[0]
                if feat_fid.size(2) != 1 or feat_fid.size(3) != 1:
                    feat_fid = F.adaptive_avg_pool2d(feat_fid, output_size=(1, 1))
                fid_act.append(feat_fid.view(feat_fid.size(0), -1).cpu().numpy())

                # --- CMMD features ---
                if self.clip_model:
                    imgs_nhwc = imgs_01.permute(0, 2, 3, 1).cpu().numpy()
                    feat_cmmd = self.clip_model.embed(imgs_nhwc).numpy()
                    cmmd_act.append(feat_cmmd)

        fid_features = np.concatenate(fid_act, axis=0) if fid_act else None
        cmmd_features = np.concatenate(cmmd_act, axis=0) if cmmd_act else None
        return fid_features, cmmd_features

    @staticmethod
    def get_stats(activations):
        if activations is None:
            return None, None
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def calculate_fid(self, mu_gen, sigma_gen, mu_target, sigma_target):
        if not self.compute_metrics:
            return 0.0
        return calculate_frechet_distance(mu_gen, sigma_gen, mu_target, sigma_target)

    def calculate_cmmd(self, target_cmmd_act, gen_cmmd_act):
        if not self.compute_metrics or self.clip_model is None:
            return 0.0
        return self.mmd_func(target_cmmd_act, gen_cmmd_act).item()
