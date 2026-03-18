import torch, torch.nn as nn
import torchvision.models as models
 
DR_GRADES={0:'No DR',1:'Mild',2:'Moderate',3:'Severe',4:'Proliferative DR'}
 
class DRClassifier(nn.Module):
    def __init__(self, backbone='resnet50', n_cls=5):
        super().__init__()
        bb=models.resnet50(pretrained=False)
        self.features=nn.Sequential(*list(bb.children())[:-1],nn.Flatten())
        self.classifier=nn.Sequential(
            nn.Linear(2048,512),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,n_cls))
    def forward(self,x): return self.classifier(self.features(x))
 
class EnsembleDR(nn.Module):
    def __init__(self, n_models=3):
        super().__init__()
        self.models=nn.ModuleList([DRClassifier() for _ in range(n_models)])
    def forward(self,x):
       probs=[torch.softmax(m(x),-1) for m in self.models]
        return torch.stack(probs).mean(0)
 
def kappa_score(y_true, y_pred, n_cls=5):
    """Quadratic weighted kappa for ordinal classification."""
    import numpy as np
    W=np.zeros((n_cls,n_cls)); O=np.zeros((n_cls,n_cls))
    for i in range(n_cls):
        for j in range(n_cls): W[i,j]=(i-j)**2/(n_cls-1)**2
    for yt,yp in zip(y_true,y_pred): O[yt,yp]+=1
    O=O/O.sum()
    hist_t=O.sum(1); hist_p=O.sum(0)
    E=np.outer(hist_t,hist_p)
    return 1-np.sum(W*O)/np.sum(W*E)
 
ensemble=EnsembleDR(3); x=torch.randn(4,3,256,256)
probs=ensemble(x); preds=probs.argmax(-1)
print(f"Input: {x.shape} → Probs: {probs.shape}")
for i,p in enumerate(preds): print(f"  Image {i}: {DR_GRADES[p.item()]} (conf={probs[i,p].item():.2f})")
import numpy as np
y_t=[0,1,2,3,4,0,1,2]; y_p=[0,1,1,3,4,1,1,2]
print(f"Quadratic Kappa: {kappa_score(y_t,y_p):.3f}")
