import torch
from aide.immunization import masking_loss
from aide.psm import fuse_conditions

def test_masking_zero():
    x=torch.randn(1,1,16000); d=torch.zeros_like(x); assert torch.isfinite(masking_loss(x,d,.8))

def test_fusion_endpoints():
    c=torch.ones(2,1,8); p=torch.zeros_like(c); s=torch.zeros(2); g=torch.ones(2,8)
    assert torch.allclose(fuse_conditions(c,p,s,g),c)
    s=torch.ones(2); assert torch.allclose(fuse_conditions(c,p,s,g),p)
