from pathlib import Path

import control as ct
import matplotlib.pyplot as plt
import numpy as np


# 电机参数
J = 0.01       # 转动惯量
b = 0.1        # 黏性阻尼系数
K = 0.01       # 电机转矩常数/反电动势常数
R = 1.0        # 电枢电阻
L = 0.5        # 电枢电感


def create_motor_model():
    """建立从输入电压到电机转速的传递函数。"""
    numerator = [K]

    denominator = [
        J * L,
        J * R + b * L,
        b * R + K**2,
    ]

    return ct.TransferFunction(numerator, denominator)


def create_pid(kp, ki, kd, filter_time=0.02):
    """建立带一阶微分滤波器的 PID 控制器。"""
    s = ct.TransferFunction.s

    proportional = kp
    integral = ki / s
    derivative = kd * s / (filter_time * s + 1)

    return proportional + integral + derivative


def simulate_controller(motor, name, kp, ki, kd, time):
    """仿真指定控制器并返回响应。"""
    controller = create_pid(kp, ki, kd)
    closed_loop = ct.feedback(controller * motor, 1)

    response_time, response = ct.step_response(
        closed_loop,
        T=time,
    )

    response = np.squeeze(response)
    information = ct.step_info(closed_loop)

    print(f"\n{name}")
    print(f"Kp={kp}, Ki={ki}, Kd={kd}")
    print(f"上升时间: {information['RiseTime']:.4f} s")
    print(f"调节时间: {information['SettlingTime']:.4f} s")
    print(f"超调量: {information['Overshoot']:.2f} %")
    print(f"稳态值: {information['SteadyStateValue']:.4f}")

    return response_time, response


def main():
    motor = create_motor_model()
    time = np.linspace(0, 10, 2000)

    controllers = [
        ("P controller", 10.0, 0.0, 0.0),
        ("PI controller", 10.0, 5.0, 0.0),
        ("PID controller", 10.0, 5.0, 0.2),
    ]

    plt.figure(figsize=(10, 6))

    for name, kp, ki, kd in controllers:
        response_time, response = simulate_controller(
            motor=motor,
            name=name,
            kp=kp,
            ki=ki,
            kd=kd,
            time=time,
        )

        plt.plot(response_time, response, label=name)

    plt.axhline(
        y=1.0,
        color="black",
        linestyle="--",
        label="Reference",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Normalized motor speed")
    plt.title("DC Motor Speed Control: P vs PI vs PID")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_directory = Path("results")
    output_directory.mkdir(exist_ok=True)

    output_file = output_directory / "pid_comparison.png"
    plt.savefig(output_file, dpi=200)

    print(f"\n结果已经保存到：{output_file}")

    plt.show()


if __name__ == "__main__":
    main()