import math

class Motor:
    """
    ： 电机类，存储电机参数，计算电机控制电流

    ： kp 电机kp参数

    ： kd 电机kd参数

    ： kt 电机时间常数 N.m/A

    ： can_id 点击的can ID

    ： 计算公式 motor_current = ((kp * (target_q - current_q) + kd * (target_v - current_v)) / (reduction_rati0 * kt)
    
    ： reduction_ratio电机减速比

    ： max_current电机最大电流
    """
    def __init__(self, kp, kd, kt, can_id, reduction_ratio, max_current):
        self.kp = kp
        self.kd = kd
        self.kt = kt
        self.can_id = can_id
        self.reduction_ratio = reduction_ratio
        self.max_current = max_current

    def calculate_motor_current(self, target_q, current_q, current_v):
        """
        计算需要发送给电机电流控制指令
        - target_q 目标位置rad
        - curret_q 当前位置rad
        - current_v 当前速度rad/s
        
        return
        - 目标电流(单位A)
        """
        # target_q rad, current_v rad/s
        motor_current = max(-self.max_current, 
                            min(self.max_current, 
                                (self.kp * (target_q - current_q) - 
                                 self.kd * current_v) / 
                                 (self.kt * self.reduction_ratio)
                                ))
        return motor_current

    def initial_motor(self, min_pos, max_pos, velocity_limit, current_limit, can):
        """
        min_pos : 位置低限位
        max_pose : 位置高限位
        velocity_limit : 速度限位 (-velocity_limit, velocity_limit)
        current_limit : 电流限制(current_limit)
        can : can的实例化对象
        """
        # 1. 发送初始化命令,回0
        can.send_command(self.can_id, [0x1E, 0x00, 0x00, 0x00, 0x00])
        # 2. 开启电流环/速度环限制 (bit0=1开启速度环位置限制)
        can.send_command(self.can_id, [0x55, 0x01, 0x00, 0x00, 0x00])
        
        # 3. 设置电流限位 (单位转换: A -> mA)
        # 电流限制值转换为mA单位
        current_limit_positive = int(current_limit * 1000)  # 转换为mA
        current_limit_negative = -current_limit_positive   # 负向电流
        
        # 转换为小端序字节(低位在前)
        current_pos_bytes = list(current_limit_positive.to_bytes(4, 'little', signed=True))
        current_neg_bytes = list(current_limit_negative.to_bytes(4, 'little', signed=True))
        
        # 发送电流限位命令
        can.send_command(self.can_id, [0x20] + current_pos_bytes)  # 最大正电流
        can.send_command(self.can_id, [0x21] + current_neg_bytes)  # 最小负电流

        # 3. 设置位置限位 (单位转换: 度 -> 协议要求的整数值)
        # 公式: (目标角度/360) * 减速比 * 65536
        max_pos_value = int((max_pos / 360.0) * self.reduction_ratio * 65536)
        min_pos_value = int((min_pos / 360.0) * self.reduction_ratio * 65536)
        # 转换为小端序字节(低位在前)
        max_pos_bytes = list(max_pos_value.to_bytes(4, 'little', signed=True))
        min_pos_bytes = list(min_pos_value.to_bytes(4, 'little', signed=True))
        # 发送位置限位命令
        can.send_command(self.can_id, [0x26] + max_pos_bytes)  # 最大正向位置
        can.send_command(self.can_id, [0x27] + min_pos_bytes)  # 最小负向位置
        
        # 4. 设置速度限位 (单位转换: rad/s -> 度/s -> 协议要求的整数值)
        # 公式: (目标转速(度/秒) * 减速比 * 100) / 360
        # 先将rad/s转换为度/s: deg_per_s = rad/s * (180/π)
        deg_per_s = velocity_limit * (180 / math.pi)
        velocity_value_positive = int((deg_per_s * self.reduction_ratio * 100) / 360)
        velocity_value_negative = -velocity_value_positive  # 负向速度
        # 转换为小端序字节(低位在前)
        vel_pos_bytes = list(velocity_value_positive.to_bytes(4, 'little', signed=True))
        vel_neg_bytes = list(velocity_value_negative.to_bytes(4, 'little', signed=True))
        # 发送速度限位命令
        can.send_command(self.can_id, [0x24] + vel_pos_bytes)  # 最大正向速度
        can.send_command(self.can_id, [0x25] + vel_neg_bytes)  # 最小负向速度
        

