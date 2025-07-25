# import torch
# import torch.nn.functional as F
# from collections import defaultdict
# from typing import Tuple

# # 假设这些变量已经定义
# spect_audio_shape = torch.Size([1, 3, 224, 224])  # 示例形状，根据实际情况修改
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def training_step(carrier: torch.Tensor, carrier_reconst: torch.Tensor, msg: torch.Tensor, msg_reconst: torch.Tensor, lambda_carrier, lambda_msg, loss_type) -> Tuple[torch.Tensor, defaultdict]:
#     try:
#         assert carrier.shape == carrier_reconst.shape == msg.shape == msg_reconst.shape == spect_audio_shape
#     except AssertionError:
#         print(f"carrier.shape:{carrier.shape},carrier_reconst.shape:{carrier_reconst.shape},msg.shape:{msg.shape},msg_reconst.shape:{msg_reconst.shape},spect_audio_shape:{spect_audio_shape}")
#     losses_log = defaultdict(int)
#     carrier, msg = carrier.to(device), msg.to(device)
#     loss = F.mse_loss if loss_type == 'mse' else F.l1_loss
#     carrier_loss = loss(carrier_reconst, carrier)
#     msg_loss = loss(msg_reconst, msg)
#     losses_log['carrier_loss'] = carrier_loss.item()
#     losses_log['msg_loss'] = msg_loss.item()
#     total_loss = lambda_carrier * carrier_loss + lambda_msg * msg_loss

#     return total_loss, losses_log

# # 示例输入数据（根据实际情况修改）
# carrier = torch.randn(spect_audio_shape)
# carrier_reconst = torch.randn(spect_audio_shape)
# msg = torch.randn(spect_audio_shape)
# msg_reconst = torch.randn(spect_audio_shape)

# # 定义要尝试的lambda值列表
# lambda_carrier_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # 示例值
# lambda_msg_values = [0.1, 0.5, 1.0, 2.0, 5.0]      # 示例值

# # 遍历不同的lambda组合
# for lambda_carrier in lambda_carrier_values:
#     for lambda_msg in lambda_msg_values:
#         # 运行训练步骤
#         loss, losses_log = training_step(carrier, carrier_reconst, msg, msg_reconst, lambda_carrier, lambda_msg, 'mse')
        
#         # 打印结果
#         print(f"lambda_carrier: {lambda_carrier}, lambda_msg: {lambda_msg}")
#         print(f"Total Loss: {loss.item()}")
#         print(f"Carrier Loss: {losses_log['carrier_loss']}, Msg Loss: {losses_log['msg_loss']}")
#         print("-" * 50)

# # 释放缓存（如果有需要）
# torch.cuda.empty_cache()