# 项目概述
    任务内容：输入部分人体表点云，预测全人体体素及类别。
    A项目为 @HyperbodyV1 , 是一个3DUnet+Hyperbolic模块的方法。
    B项目为 @nnUnet ， 是经典的3d医学图像分割算法

# 环境选择
    针对项目 HyperbodyV1: conda activate pasco
    针对项目 nnUnet: conda activate nnunet
# 可视化
    - 对于3D内容，尽量选择使用plotly，绘制可交互的html格式结果
    - 针对普通图表，选择PNG格式图片
# TDD 测试
    - 添加任何新功能前，首先完成测试脚本书写。
    - 多可视化，以及检查tensor形状的测试，让测试结果清晰可见。
    - 每次测试后，需要汇a.报改动文件是什么；b.测试结果
# 语言问题
    写代码期间，完全使用英语，包括注释、print中的提示等
# 计算效率问题
    在大计算量的问题上，慎重使用python for的逻辑。尽可能使用向量的方法一次性计算。
# 维度命名规则
    (B, C, D, H, W ) 
    B = Batch size (批次大小)
    C = Channels (通道数：num_classes, embed_dim 等)
    D = Depth (空间深度，Z轴方向)
    H = Height (高度)
    W = Width (宽度)
