"""CPU-only mocked boundary test for the Comfy adapter (no Iris weights)."""
import importlib.util
import sys
import types
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import torch

p=Path(__file__).with_name('nodes.py')
folder=types.ModuleType('folder_paths')
folder.get_filename_list=lambda typ:['iris.safetensors']
folder.get_full_path_or_raise=lambda typ,name:'/tmp/iris.safetensors'
sys.modules['folder_paths']=folder
iris=types.ModuleType('iris3b'); iris.__path__=[]
text=types.ModuleType('iris3b.text'); text.__path__=[]
base=types.ModuleType('iris3b.text.base')
class TextEncoding:
    def __init__(self,embeddings,mask): self.embeddings,self.mask=embeddings,mask
base.TextEncoding=TextEncoding
sampling=types.ModuleType('iris3b.sampling')
observed={}
def fake_generate(model,encoder,prompts,**kwargs):
    observed['prompt']=prompts
    observed['cfg']=kwargs['cfg_scale']
    observed['shift']=kwargs['shift']
    observed['steps']=kwargs['steps']
    observed['pos_shape']=tuple(encoder.encode(prompts).embeddings.shape)
    observed['neg_shape']=tuple(encoder.null('').embeddings.shape)
    return torch.zeros(1,3,16,16)
sampling.generate=fake_generate
for name,obj in [('iris3b',iris),('iris3b.text',text),('iris3b.text.base',base),('iris3b.sampling',sampling)]:
    sys.modules[name]=obj
spec=importlib.util.spec_from_file_location('iris_adapter_nodes',p)
node=importlib.util.module_from_spec(spec); spec.loader.exec_module(node)
config=SimpleNamespace(model=SimpleNamespace(patch_size=16),
    sample=SimpleNamespace(cfg_interval=(0.,1.)),
    flow=SimpleNamespace(num_train_timesteps=1000,prediction='v'))
class FakeModel(torch.nn.Module): pass
node._source=lambda:None
node._config=lambda p:config
node._build_model=lambda *args,**kwargs:(FakeModel(),False)
node._cleanup=lambda:None
ckpt=SimpleNamespace(stat=lambda:SimpleNamespace(st_mtime_ns=99))
cond={'positive':torch.ones(1,300,12,2560), 'positive_mask':torch.ones(1,300,dtype=torch.int64),
      'negative':torch.zeros(1,300,12,2560),'negative_mask':torch.ones(1,300,dtype=torch.int64),
      'prompt':'fox in snow','negative_prompt':''}
orig_generator=torch.Generator
with patch.object(torch.cuda,'is_available',return_value=True), \
     patch.object(torch,'autocast',return_value=nullcontext()), \
     patch.object(torch,'Generator',side_effect=lambda device:orig_generator(device='cpu')):
    (image,) = node.Iris3BSampler().sample(
      {'path':ckpt},cond,16,16,20,2,3.,4.,123,'bfloat16','native_quantized',True)
assert tuple(image.shape)==(1,16,16,3)
assert float(image.min())==float(image.max())==0.5
assert observed=={'prompt':['fox in snow'],'cfg':3.,'shift':4.,'steps':20,
                  'pos_shape':(1,300,12,2560),'neg_shape':(1,300,12,2560)}
print('PASS — official generate entrypoint receives exact prompt, encoding tensors, CFG, shift, steps')
print('PASS — returned pixel image NCHW [-1,1] correctly converted to Comfy NHWC [0,1]')
