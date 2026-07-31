import sys, os, torch
sys.path.insert(0, './')
from easydict import EasyDict
from stereo.utils import common_utils
from stereo.modeling.models.dbstereo.dbstereo import DBStereo

CKPT_CFGS = [
    ('ckpts/dbstereo_m_sceneflow.pth', 'cfgs/dbstereo/dbstereo_m_sceneflow.yaml'),
    ('ckpts/dbstereo_l_sceneflow.pth', 'cfgs/dbstereo/dbstereo_l_sceneflow.yaml'),
]

for ckpt_path, cfg_path in CKPT_CFGS:
    print('\n' + '=' * 70)
    print(f'Checking: {ckpt_path}  (cfg: {cfg_path})')
    yaml = common_utils.config_loader(cfg_path)
    cfgs = EasyDict(yaml)

    # mimic the auto-detect logic in trainer_template.build_model
    model = DBStereo(cfgs.MODEL)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_state = ckpt.get('model_state', ckpt)
    has_conv_final = any('geo_decoder.conv_final.weight' in k for k in ckpt_state)
    if not has_conv_final and isinstance(model.geo_decoder.conv_final, torch.nn.Conv2d):
        model.geo_decoder.conv_final = torch.nn.Identity()

    model_sd = model.state_dict()
    unused, unupdate = [], []
    for k, v in ckpt_state.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            continue
        unused.append((k, tuple(v.shape)))
    for k in model_sd:
        if k not in ckpt_state:
            unupdate.append((k, tuple(model_sd[k].shape)))

    print(f'  checkpoint keys: {len(ckpt_state)}  model keys: {len(model_sd)}')
    print(f'  UNUSED (in ckpt, not in model / shape mismatch): {len(unused)}')
    for k, s in unused[:20]:
        print(f'    {k}: {s}')
    print(f'  NOT UPDATED (in model, not in ckpt): {len(unupdate)}')
    for k, s in unupdate[:20]:
        print(f'    {k}: {s}')
    ok = (len(unused) == 0 and len(unupdate) == 0)
    print(f'  => {"FULLY LOADABLE" if ok else "MISMATCH"}')
