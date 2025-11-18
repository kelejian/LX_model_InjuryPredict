"""
通用化模型测试函数,用于查看模型结构、检测是否正常前向和反向传播：
1. 接受任意模型实例化对象 `model`。
2. 自定义输入 `inputs` 和标签 `labels`。
3. 支持前向传播、反向传播、损失计算。
4. 导出 ONNX 模型并验证。
5. 输出模型详细信息。
参数：
- model: PyTorch 模型实例化对象: torch.nn.Module
- inputs: 模型的输入张量: tensor 或 tulple(tensor1, tensor2, ...) 或 list(tensor1, tensor2, ...)
- labels: 模型的真实标签张量（用于损失计算）: tensor
- criterion: 损失函数实例化对象，默认为 nn.MSELoss
- optimizer: 优化器实例化对象，默认为 Adam
- onnx_file_path: 导出的 ONNX 文件路径
"""
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.onnx
from torchinfo import summary
from torchviz import make_dot

# @profile
def test_model(
    model,
    inputs,
    labels,
    criterion=None,
    optimizer=None,
    onnx_file_path="model_test.onnx"
):
    """
    通用化模型测试函数：
    1. 接受任意模型实例化对象 `model`。
    2. 自定义输入 `inputs` 和标签 `labels`。
    3. 支持前向传播、反向传播、损失计算。
    4. 导出 ONNX 模型并验证。
    5. 输出模型详细信息。
    
    参数：
    - model: PyTorch 模型实例化对象: torch.nn.Module
    - inputs: 模型的输入张量: tensor 或 tulple(tensor1, tensor2, ...) 或 list(tensor1, tensor2, ...)
    - labels: 模型的真实标签张量（用于损失计算）: tensor
    - criterion: 损失函数实例化对象，默认为 nn.MSELoss
    - optimizer: 优化器实例化对象，默认为 Adam
    - onnx_file_path: 导出的 ONNX 文件路径
    """
    # 默认损失函数和优化器
    if criterion is None:
        criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 将模型设为训练模式
    model.train()
    print("\n~~~~~~~~~~~~~~~~~~~ 🚀🚀 开始测试神经网络模型是否可以正常训练 🚀🚀 ~~~~~~~~~~~~~~~~~~~~")
    # 打印模型结构信息
    print("\n============== 模型结构信息 ==============")
    _input_data = tuple(inputs) if isinstance(inputs, (tuple, list)) else inputs
    summary(
        model,
        input_data=_input_data,
        col_names=["input_size", "output_size", "num_params"],
        depth=3,
        device="cuda" if next(model.parameters()).is_cuda else "cpu"
    )
    
    # 前向传播与loss计算
    print("\n============== 前向传播 ==============")
    if isinstance(inputs, (tuple, list)):
        outputs = model(*inputs)
        # 一行打印模型各个输入input的形状
        print(f"✔ 模型各个输入的形状：{[input.shape for input in inputs]}")

    else:
        outputs = model(inputs)
        print(f"✔ 输入形状：{inputs.shape}")

    # 初始化loss变量
    loss = None
    
    if isinstance(outputs, (tuple, list)):
        print(f"✔ 模型各个输出的形状：{[output.shape for output in outputs]}")
        for i, output in enumerate(outputs):
            if labels.shape == output.shape:
                loss = criterion(output, labels)
                print(f"✔ 第{i+1}个模型输出对应了一个loss值: {loss.item()}")
                break  # 找到第一个匹配的输出就停止
        
        if loss is None:
            print("✘ 没有找到与标签形状匹配的输出，使用第一个输出计算损失")
            loss = criterion(outputs[0], labels)
    else:
        print(f"✔ 模型输出形状：{outputs.shape}")
        if labels.shape == outputs.shape:
            loss = criterion(outputs, labels)
            print(f"✔ 损失值：{loss.item()}")
        else: 
            print("✘ 模型输出形状与标签形状不匹配，尝试计算损失值")
            # 尝试计算损失，即使形状不完全匹配
            try:
                loss = criterion(outputs, labels)
                print(f"✔ 强制计算的损失值：{loss.item()}")
            except Exception as e:
                print(f"✘ 无法计算损失值: {e}")
                return  # 如果无法计算损失，提前返回

    # 确保loss不为None
    if loss is None:
        print("✘ 无法获得有效的loss值，停止测试")
        return

    # 反向传播
    print("\n============== 反向传播 ==============")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("✔ 反向传播正常~")

    # 可视化计算图
    print("\n============== 计算图可视化 ==============")
    graph = make_dot(loss, params=dict(model.named_parameters()))
    graph.render("model_computation_graph", format="png")
    print("✔ 计算图已保存为 'model_computation_graph.png'")

    # 导出 ONNX 模型
    print("\n============== 导出 ONNX 模型 ==============")
    
    # 根据输入类型配置输入名称和动态轴
    if isinstance(inputs, (tuple, list)):
        input_names = [f"input_{i}" for i in range(len(inputs))]
        dynamic_axes = {f"input_{i}": {0: "batch_size"} for i in range(len(inputs))}
    else:
        input_names = ["input"]
        dynamic_axes = {"input": {0: "batch_size"}}
    
    # 配置输出名称和动态轴
    if isinstance(outputs, (tuple, list)):
        output_names = [f"output_{i}" for i in range(len(outputs))]
        for i in range(len(outputs)):
            dynamic_axes[f"output_{i}"] = {0: "batch_size"}
    else:
        output_names = ["output"]
        dynamic_axes["output"] = {0: "batch_size"}
    
    torch.onnx.export(
        model,
        _input_data,
        onnx_file_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=11,
    )
    print(f"✔ ONNX 模型已保存至 {onnx_file_path}")
    print("在 https://netron.app/ 上查看 ONNX 模型结构")

    # # 使用 ONNX Runtime 推理
    # print("\n============== ONNX Runtime 推理 ==============")
    # ort_session = onnxruntime.InferenceSession(onnx_file_path)
    # ort_inputs = {
    #     onnx_model.graph.input[i].name: (
    #         inputs[i].cpu().numpy() if isinstance(inputs, (tuple, list))
    #         else inputs.cpu().numpy()
    #     )
    #     for i in range(len(onnx_model.graph.input))
    # }
    # ort_outs = ort_session.run(None, ort_inputs)
    # print(f"ONNX 推理输出：{ort_outs}")

if __name__ == "__main__":
    import os
    import numpy as np
    from utils.dataset_prepare import CrashDataset
    from utils.models import TeacherModel, StudentModel
    from utils.weighted_loss import weighted_loss
    
    train_dataset = torch.load(os.path.join("data", "train_dataset.pt"))

    # 定义模型相关的超参数
    
    Ksize_init = 8 # TCN 初始卷积核大小，必须是偶数 4-12
    Ksize_mid = 5  # TCN 中间卷积核大小，必须是奇数 3 or 5
    num_blocks_of_tcn = 3  # TCN 的块数 2 - 6
    tcn_channels_list = [64, 128, 256]  # 每个 TCN 块的输出通道数列表
    num_layers_of_mlpE = 3  # MLP 编码器的层数 4-5
    num_layers_of_mlpD = 3  # MLP 解码器的层数 4-5
    mlpE_hidden = 224  # MLP 编码器的隐藏层维度 96 - 192
    mlpD_hidden = 160  # MLP 解码器的隐藏层维度 128 or 256
    encoder_output_dim = 96  # 编码器输出特征维度 64 or 96
    decoder_output_dim = 16  # 解码器输出特征维度 16 or 32 or 64
    dropout_TCN = 0.15  # TCN Dropout 概率 0.05-0.15
    dropout_MLP = 0.20  # Dropout 概率 0.05-0.25
    use_channel_attention=True  # 是否使用注意力机制
    fixed_channel_weight = [0.6, 0.4, 0]  # 固定的通道注意力权重(None表示自适应学习)

    # 将模型移动到CUDA设备
    # 加载模型
    model = TeacherModel(
        Ksize_init=Ksize_init,
        Ksize_mid=Ksize_mid,
        num_classes_of_discrete=train_dataset.dataset.num_classes_of_discrete, # --- 修改：从加载的训练集中获取元数据 ---
        num_blocks_of_tcn=num_blocks_of_tcn,
        tcn_channels_list=tcn_channels_list,
        num_layers_of_mlpE=num_layers_of_mlpE,
        num_layers_of_mlpD=num_layers_of_mlpD,
        mlpE_hidden=mlpE_hidden,
        mlpD_hidden=mlpD_hidden,
        encoder_output_dim=encoder_output_dim,
        decoder_output_dim=decoder_output_dim,
        dropout_MLP=dropout_MLP,
        dropout_TCN=dropout_TCN,
        use_channel_attention=use_channel_attention,
        fixed_channel_weight=fixed_channel_weight
    )

    num_layers_of_mlpE = 3  # MLP 编码器的层数
    num_layers_of_mlpD = 3  # MLP 解码器的层数
    mlpE_hidden = 224  # MLP 编码器的隐藏层维度
    mlpD_hidden = 160  # MLP 解码器的隐藏层维度
    encoder_output_dim = 96  # 编码器输出特征维度
    decoder_output_dim = 16  # 解码器输出特征维度
    dropout = 0.15  # Dropout 概率


    # model = StudentModel(
    #     num_classes_of_discrete=dataset.num_classes_of_discrete,
    #     num_layers_of_mlpE=num_layers_of_mlpE, num_layers_of_mlpD=num_layers_of_mlpD,
    #     mlpE_hidden=mlpE_hidden, mlpD_hidden=mlpD_hidden,
    #     encoder_output_dim=encoder_output_dim, decoder_output_dim=decoder_output_dim,
    #     dropout=dropout
    # )

    # model移动到CUDA
    model = model.cuda()

    # 示例输入数据（模拟数据集第1个batch）
    batch_size = 128

    x_acc = torch.tensor(train_dataset.dataset.x_acc[:batch_size], dtype=torch.float32).cuda()  # (B, 3, 150)
    x_att_con = torch.tensor(train_dataset.dataset.x_att_continuous[:batch_size], dtype=torch.float32).cuda()  # (B, 14)
    x_att_dis = torch.tensor(train_dataset.dataset.x_att_discrete[:batch_size], dtype=torch.long).cuda()  # (B, 4)
    y_HIC = torch.tensor(train_dataset.dataset.y_HIC[:batch_size], dtype=torch.float32).cuda() # (B,)
    y_Dmax = torch.tensor(train_dataset.dataset.y_Dmax[:batch_size], dtype=torch.float32).cuda() # (B,)
    y_Nij = torch.tensor(train_dataset.dataset.y_Nij[:batch_size], dtype=torch.float32).cuda() # (B,)
    y = torch.stack([y_HIC, y_Dmax, y_Nij], dim=1)  # (B, 3)

    criterion = weighted_loss()
    # 测试模型
    test_model(model, inputs=(x_acc, x_att_con, x_att_dis), labels=y)
    #test_model(model, inputs=(x_att_con, x_att_dis), labels=y)
