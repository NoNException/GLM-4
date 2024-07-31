import os
import json
import shutil
from modelscope.metainfo import Metrics, Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile
from modelscope.hub.snapshot_download import snapshot_download

WORKSPACE = "./workspace"
ocr_model = 'damo/ofa_ocr-recognition_scene_base_zh'
# 下载模型至缓存目录，并返回目录, 这里是模型输出目录
ocr_path = snapshot_download(ocr_model)
# ofa通用的pretrained模型，未针对OCR场景做过调优# 预训练模型的模型id
pretrained_model = 'damo/ofa_pretrain_base_zh'
# 下载预训练模型tag时间低于modelscope v1.0.2的发布时间，所以使用ms 1.0.2版本时需要额外增加具体的tag version
pretrained_path = snapshot_download(pretrained_model, revision='v1.0.0')
shutil.copy(os.path.join(ocr_path, ModelFile.CONFIGURATION),  # 将任务的配置覆盖预训练模型的配置
            os.path.join(pretrained_path, ModelFile.CONFIGURATION))
os.makedirs(WORKSPACE, exist_ok=True)
# 写一下配置文件
config_file = os.path.join(WORKSPACE, ModelFile.CONFIGURATION)
with open(config_file, 'w') as writer:
    json.dump(finetune_cfg, writer, indent="\t")
# trainer的其他配置项
args = dict(
    model=pretrained_path,  # 要继续finetune的模型
    work_dir=WORKSPACE,
    train_dataset=MsDataset.load(  # 数据集，这里msdataset兼容huggingface的dataset
        'ocr_fudanvi_zh',  # msdataset的id
        namespace='modelscope',
        split='train'),
    eval_dataset=MsDataset.load('ocr_fudanvi_zh', namespace='modelscope', split='validation'),
    cfg_file=config_file)  # 配置文件地址
trainer = build_trainer(name=Trainers.ofa, default_args=args)  # 构建训练器
trainer.train()
