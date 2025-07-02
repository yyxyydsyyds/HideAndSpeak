import rclpy
from rclpy.node import Node
from ros2robot_interfaces.msg import HumanoidAction, MotorDebug
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from ctypes import *
from motor_can.lib.can_utils import CANController
from motor_can.lib.can_utils import *
from motor_can.lib.motor import Motor
import threading
import math
import time

class MotorControlNode(Node):
    def __init__(self):
        super().__init__('motor_controller')
        
        # 电机配置参数
        self.motor_configs = {
            'left_femur1': {'kp': 30, 'kd': 1.5, 'kt': 0.143, 'can_id': 0x01, 'reduction_ratio': 51, 'max_current': 1.5},
            'left_femur2': {'kp': 30, 'kd': 1.5, 'kt': 0.143, 'can_id': 0x02, 'reduction_ratio': 51, 'max_current': 1.5},
            'left_upper_leg': {'kp': 30, 'kd': 1.5, 'kt': 0.143, 'can_id': 0x03, 'reduction_ratio': 51, 'max_current': 1.5},
            'left_lower_leg': {'kp': 30, 'kd': 1.5, 'kt': 0.118, 'can_id': 0x04, 'reduction_ratio': 81, 'max_current': 1.5},
            'left_foot': {'kp': 30, 'kd': 1.5, 'kt': 0.089, 'can_id': 0x05, 'reduction_ratio': 81, 'max_current': 1.5},
            'right_femur1': {'kp': 30, 'kd': 1.5, 'kt': 0.143, 'can_id': 0x06, 'reduction_ratio': 51, 'max_current': 1.5},
            'right_femur2': {'kp': 30, 'kd': 1.5, 'kt': 0.143, 'can_id': 0x07, 'reduction_ratio': 51, 'max_current': 1.5},
            'right_upper_leg': {'kp': 30, 'kd': 1.5, 'kt': 0.143, 'can_id': 0x08, 'reduction_ratio': 51, 'max_current': 1.5},
            'right_lower_leg': {'kp': 30, 'kd': 1.5, 'kt': 0.118, 'can_id': 0x09, 'reduction_ratio': 81, 'max_current': 1.5},
            'right_foot': {'kp': 30, 'kd': 1.5, 'kt': 0.089, 'can_id': 0x10, 'reduction_ratio': 81, 'max_current': 1.5}
        }
        
        # 初始化电机
        self.motors = {name: Motor(**config) for name, config in self.motor_configs.items()}
        
        # 初始化CAN控制器
        self.can = CANController()
        
        # 动作和状态字典
        self.actions = {name: 0.0 for name in self.motor_configs}
        self.states = {}
        for name in self.motor_configs:
            self.states[f"{name}_q"] = 0.0  # 位置
            self.states[f"{name}_v"] = 0.0  # 速度
            self.states[f"{name}_i"] = 0.0  # 电流

        # 初始化电机状态
        self._init_motor_state()

        # 创建订阅者
        self.subscription = self.create_subscription(
            HumanoidAction, 
            '/action',
            self.action_callback,
            10)
        
        # 新增调试发布者 (发布Float64MultiArray消息)
        self.debug_pub = self.create_publisher(
            MotorDebug, 
            '/motor_debug', 
            10
        )

        # 创建发布者
        self.publisher = self.create_publisher(JointState, '/joint_state', 10)
        
        # 创建定时器
        self.timer = self.create_timer(0.002, self.motor_controller_callback)
        
    def _init_motor_state(self):
        # 电机初始化参数
        init_params = {
            'left_femur1': {'min_pos': -35, 'max_pos': 35, 'current_limit' : 1.5},
            'left_femur2': {'min_pos': -20, 'max_pos': 20, 'current_limit' : 3},
            'left_upper_leg': {'min_pos': -35, 'max_pos': 35, 'current_limit' : 3},
            'left_lower_leg': {'min_pos': -25, 'max_pos': 15, 'current_limit' : 3},
            'left_foot': {'min_pos': -15, 'max_pos': 25, 'current_limit' : 3},
            'right_femur1': {'min_pos': -35, 'max_pos': 35, 'current_limit' : 1.5},
            'right_femur2': {'min_pos': -20, 'max_pos': 20, 'current_limit' : 3},
            'right_upper_leg': {'min_pos': -35, 'max_pos': 35, 'current_limit' : 3},
            'right_lower_leg': {'min_pos': -15, 'max_pos': 25, 'current_limit' : 3},
            'right_foot': {'min_pos': -25, 'max_pos': 15, 'current_limit' : 3}
        }
        
        for name, motor in self.motors.items():
            params = init_params[name]
            motor.initial_motor(
                min_pos=params['min_pos'],
                max_pos=params['max_pos'],
                velocity_limit=1,  #回0限速
                current_limit=params['current_limit'],
                can=self.can
            )
        time.sleep(3)


    def action_callback(self, msg):
        """更新目标位置"""
        for name in self.motor_configs:
            self.actions[name] = getattr(msg, name)

    def motor_controller_callback(self):
        """电机控制回调"""
        # 准备发布关节状态
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = self.get_clock().now().to_msg()

        for name, motor in self.motors.items():
            q = self.states[f"{name}_q"]
            v = self.states[f"{name}_v"]
                
            # 添加到发布消息
            joint_state.name.append(name)
            joint_state.position.append(q)
            joint_state.velocity.append(v)

            # 计算电机电流
            current = motor.calculate_motor_current(self.actions[name], q, v)
                
            # 转换为毫安并发送
            current_int = int(round(current * 1000))
            current_bytes = list(current_int.to_bytes(4, 'little', signed=True))
            self.can.send_command(motor.can_id, [0x42] + current_bytes)

        #接受电机状态
        self.can_receive_loop()

        # 发布关节状态
        self.publisher.publish(joint_state)

        #发布电机调试数据
        debug_msg = MotorDebug()
        debug_msg.target_pos = self.actions['left_femur1']# 目标位置 (rad)
        debug_msg.target_i = self.motors['left_femur1'].calculate_motor_current(self.actions['left_femur1'], self.states['left_femur1_q'], self.states['left_femur1_v'])
        debug_msg.current_pos = self.states['left_femur1_q'] #实际位置
        debug_msg.current_i = self.states['left_femur1_i']  # 实际电流 (A)
        self.debug_pub.publish(debug_msg)

    def can_receive_loop(self):
        """CAN接收线程循环"""
        rx_array = VCI_CAN_OBJ_ARRAY(2500)
        for can_port in [0, 1]:
            ret = self.can.canDLL.VCI_Receive(4, 0, can_port, byref(rx_array.ADDR), 2500, 0)
            if ret > 0:
                for i in range(ret):
                    self.process_can_frame(rx_array.STRUCT_ARRAY[i])
   

    def process_can_frame(self, frame):
        """处理CAN帧"""
        if 0x01 <= frame.ID <= 0x10:
            motor_id = frame.ID

            # 找到对应的电机名称
            motor_name = next((name for name, config in self.motor_configs.items() 
                              if config['can_id'] == motor_id), None)
            if not motor_name:
                return
                
            current = int.from_bytes(bytes(frame.Data[0:2]), byteorder='little', signed=True)
            self.states[f"{motor_name}_i"] = current / 1000.0  # 转换为安培
                    
            # 解析速度 (2字节)
            speed = int.from_bytes(bytes(frame.Data[2:4]), byteorder='little', signed=True)
            ratio = self.motors[motor_name].reduction_ratio

            self.states[f"{motor_name}_v"] = (speed / 100.0 / ratio) * math.pi * 2
                    
            # 解析位置 (4字节)
            position = int.from_bytes(bytes(frame.Data[4:8]), byteorder='little', signed=True)
            self.states[f"{motor_name}_q"] = (position / 65536.0 / ratio) * math.pi * 2
                    
        
    def destroy_node(self):
        # 停止所有电机
        for motor in self.motors.values():
            self.can.send_command(motor.can_id, [0x02])
        
        # 清理CAN设备
        self.can.canDLL.VCI_CloseDevice(4, 0)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    motor_controller = MotorControlNode()
    try:
        rclpy.spin(motor_controller)
    except KeyboardInterrupt:
        pass
    finally:
        motor_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
