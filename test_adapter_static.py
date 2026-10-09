"""Static reproducibility checks; runnable without ComfyUI/CUDA/model weights."""
import ast
from pathlib import Path
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parent
src=(ROOT/'nodes.py').read_text(encoding='utf-8')
parsed=ast.parse(src)
cfg=OmegaConf.load(ROOT/'configs'/'iris3b.yaml')
assert cfg.model.depth==24 and cfg.model.dual_depth==8
assert cfg.model.patch_size==16 and cfg.model.pixel.modulation=='post'
assert cfg.model.pixel.depth==4 and cfg.model.pixel.hidden_size==16
assert cfg.flow.prediction=='v' and cfg.flow.shift==4.0
assert cfg.text_encoder.hidden_layers == [2,5,8,11,14,17,20,23,26,29,32,35]
assert cfg.model.rope_aspect=='isotropic'
func_names={f.name for f in ast.walk(parsed) if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))}
for obsolete in ('_build_prompt_batch', '_ProgressFlowDPMSolver'):
    assert obsolete not in func_names, f'Found obsolete duplicated Iris code: {obsolete}'
assert 'from iris3b.text.qwen3_vl import Qwen3VLTextEncoder' in src
assert 'from iris3b.sampling import generate as iris_generate' in src
assert 'with torch.inference_mode(), torch.autocast(' in src
assert 'folder_paths.get_filename_list("checkpoints")' in src
assert 'D:\\iris-3b' not in src
assert not any(x in src for x in ('F.unfold','F.fold','FlowDPMSolver('))
assert (ROOT/'install_iris_source.py').exists()
print('PASS — upstream encoder and sampler imports used')
print('PASS — no duplicate Iris prompt, solver, or patch reconstruction code')
print('PASS — official release config key values')
print('PASS — native ComfyUI checkpoint dropdown; no hardcoded machine path')
