2025-10-9
仅为phase1代码，如需运行，请自行创建文件data，cache，outputs

2025-10-30
1.删除eval_downstream.py，改为eval.py：将extension，squat，gait三个动作的细分项通过映射归为一类（目标是对动作大类的识别），但在预训练是保留细分项，同时删掉了Logreg_balanced和MLP256（意义不大）
2.完善data_pipeline.py：添加受试者交叉验证机制，确保训练集和测试集不存在同一受试者造成数据泄露。并对稀有动作（仅有一个受试者进行的动作）进行过滤，避免对训练和下游验证产生影响。此外保存受试者分割的切分，利于对照实验（可删除cache里的split产生新的切分）
3.更替config.yaml：epochs调整为20（20足够比较模型差异），mae_loss与order_loss改为0.5（供参考），新增subject_split_path，scale_to_unit，tenable_audit，min_subjects_per_action等参数，对应data_pipeline.py的改动
4.上传demon3.py：为phase2的下游验证，统一验证手段更能横向对比
