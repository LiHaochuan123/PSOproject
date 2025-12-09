import os  # 添加这行
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

class HorizontalWellTemperature:
    """
    水平井温度计算模型
    考虑油藏参数、井筒参数、井筒热传导、对流换热等因素
    """

    def __init__(self):
        # 水平井参数
        self.l_well = 500.0  # 水平段距离，m
        self.D_cem = 0.484  # 水泥环半径
        self.D_taoouter = 0.1778  # 套管外径，m
        self.D_taoinner = 0.1594  # 套管内径，m
        self.D_yououter = 0.1397  # 油管外径，m
        self.D_youinner = 0.1214  # 油管内径，m
        self.K_tao = 0.16  # 套管热导率，W·m^-1·K^-1
        self.K_you = 0.59  # 油管热导率，W·m^-1·K^-1
        self.K_cem = 1.73  # 水泥环导热率
        self.E_you = 1e-5  # 井壁粗糙度
        self.P_heel = 15e6  # 趾部压力，pa
        self.θ = 5   # 井筒与水平线的夹角

        # 裂缝参数
        self.N_fractures = 5  # 裂缝总数，条
        self.pos_fracture = [25, 125, 225, 325, 425]  # m，裂缝位置
        self.l_fractures = [150, 150, 150, 150, 150]  # 裂缝半长，m
        self.w_fractures = [0.15, 0.15, 0.15, 0.15, 0.15]  # 裂缝宽度，m
        self.h_fractures = [20, 20, 20, 20, 20]  # 裂缝高度，m
        self.c_fractures = [0.15, 0.15, 0.15, 0.15, 0.15]  # mD·m或者1μ㎡=1D=10^3mD,导流能力
        self.k_fractures = 1e-12  # 渗透率
        self.Qo_fractures = [21, 21, 21, 21, 21]  # 裂缝油相产量，m³/d
        self.Qw_fractures = [19, 19, 19, 19, 19]  # 裂缝水相产量，m³/d
        self.Qg_fractures = [100, 100, 100, 100, 100]  # 裂缝气相产量，m³/d

        # 流体性质
        self.ρ_o = 850.0  # 油相密度，kg/m³
        self.ρ_w = 1000.0  # 水相密度，kg/m³
        self.ρ_g = 260.0  # 气相密度，kg/m³
        self.μ_o = 5.0 * 10 ** -3  # 油相粘度，Pa·s
        self.μ_w = 1.0 * 10 ** -3  # 水相粘度，Pa·s
        self.μ_g = 0.02 * 10 ** -3  # 气相粘度，Pa·s (修正：不能为0)
        self.β_o = 0.0005  # 油相热膨胀系数，K^-1
        self.β_w = 0.0000556  # 水相热膨胀系数，K^-1
        self.β_g = 0.0015  # 气相热膨胀系数，K^-1
        self.Cp_o = 2700  # 油相比热容，J·kg^-1·K^-1
        self.Cp_w = 4200  # 水相比热容，J·kg^-1·K^-1
        self.Cp_g = 3000  # 气相比热容，J·kg^-1·K^-1
        self.K_o = 0.16  # 油相热导率，W·m^-1·K^-1
        self.K_w = 0.59  # 水相热导率，W·m^-1·K^-1
        self.K_g = 0.052  # 气相热导率，W·m^-1·K^-1
        self.g = 9.8  # 重力加速度，m/s^2

        # 油藏参数
        self.l_reservoir = 1000  # m, 油藏长度
        self.w_reservoir = 500  # m, 油藏宽度
        self.h_reservoir = 20  # m, 油藏高度
        self.K_reservoir = 5e-15  # m², 油藏渗透率
        self.Φ_reservoir = 0.15  # 孔隙度
        self.P_reservoir_initial = 30e6  # Pa, 初始压力
        self.T_reservoir_initial = 80  # °C, 初始温度

        # 岩石参数
        self.Cp_rock = 800  # J/kg·K, 岩石比热容
        self.K_rock = 2.5  # W/m·K，岩石导热系数
        self.ρ_rock = 2500  # kg/m³, 岩石密度
        self.K_a = 0.025  # W/m·K，空气导热系数

        # 打开程度
        self.η = 0.2

        # 生产参数
        self.Q_well = 200  # 水平井筒产量， m³/d

        # 初始化结果存储
        self.results = {}

    def preprocess_data(self):
        """计算过程中的所有参数"""
        print("计算过程中的所有参数...")

        # 裂缝流体参数计算
        Qo_fx = [q / (24 * 3600) for q in self.Qo_fractures]  # 裂缝油相产量，m³/s
        Qw_fx = [q / (24 * 3600) for q in self.Qw_fractures]  # 裂缝水相产量，m³/s
        Qg_fx = [q / (24 * 3600) for q in self.Qg_fractures]  # 裂缝气相产量，m³/s
        Qt_fx = np.array(Qo_fx) + np.array(Qw_fx) + np.array(Qg_fx)  # 裂缝产量，m³/s

        # 裂缝总流量
        Q_t = sum(Qt_fx)

        # 裂缝体积分数计算
        x_o = np.sum(Qo_fx) / Q_t  # 油相体积分数
        x_w = np.sum(Qw_fx) / Q_t  # 水相体积分数
        x_g = np.sum(Qg_fx) / Q_t  # 气相体积分数

        # 裂缝其他流体参数计算
        μ_fx = x_o * self.μ_o + x_w * self.μ_w + x_g * self.μ_g
        Cp_fx = x_o * self.Cp_o + x_w * self.Cp_w + x_g * self.Cp_g
        β_fx = x_o * self.β_o + x_w * self.β_w + x_g * self.β_g

        # 裂缝流体导热系数计算
        K_fl = x_o * self.K_o + x_w * self.K_w + x_g * self.K_g  # 流体导热率

        # 岩石密度计算
        ρ_rock_solid = (1 - self.Φ_reservoir) * self.ρ_rock

        # 油藏密度计算
        ρ1 = x_o * self.ρ_o + x_w * self.ρ_w + x_g * self.ρ_g  # 流体密度
        ρ_t = ρ1 * self.Φ_reservoir + ρ_rock_solid

        # 岩石与流体导热系数计算
        K1 = ((K_fl / self.K_a) ** 0.33 - 1) * 0.299
        K2 = ((self.Φ_reservoir * K_fl / (1 - self.Φ_reservoir) / self.K_a) ** 0.482) * 4.57
        K3 = (ρ_t / ρ_rock_solid) ** -4.3
        K_T = self.K_rock * (1 + K1 + K2 * K3)

        # 井筒流体参数计算
        Q_well = self.Q_well / 24 / 3600  # 井筒产量，m³/s
        xo_well = 0.14
        xw_well = 0.15
        xg_well = 0.71


        # 保存计算结果
        self.K_T = K_T
        self.K_fl = K_fl
        self.ρ_t = ρ_t
        self.ρ1 = ρ1
        self.ρ_f = ρ1  # 混合流体密度
        self.Cp_fx = Cp_fx
        self.β_fx = β_fx
        self.μ_fx = μ_fx
        self.x_o = x_o
        self.x_w = x_w
        self.x_g = x_g
        self.Q_t = Q_t
        self.Qt_fx = Qt_fx  # 每条裂缝的总流量
        self.Q1_fractures = Qt_fx.tolist()  # 用于压力计算
        self.Q_well = Q_well
        self.xo_well = xo_well
        self.xw_well = xw_well
        self.xg_well = xg_well

        return K_T, ρ_t, Cp_fx, β_fx, μ_fx, x_o, x_w, x_g, Q_t, K_fl

    def parameter(self):
        """参数计算"""
        print("计算参数...")

        # n参数计算
        a1 = self.Q_t / (4 * np.pi * self.l_well)
        a2 = self.ρ_t * (self.Cp_fx * self.Φ_reservoir + self.Cp_rock * (1 - self.Φ_reservoir)) / self.K_T
        a3 = (a2 ** 2 + 4 * self.β_fx * self.μ_fx / (self.K_reservoir * self.K_T)) ** (1 / 2)  # 修正：使用self.μ_fx
        n_p = a1 * (a2 + a3)
        n_s = a1 * (a2 - a3)

        # 保存参数
        self.n_p = n_p
        self.n_s = n_s

        return n_p, n_s

    def Inlettemperature(self):
        """周围温度、入流温度和环空温度的计算"""
        print("计算周围温度、入流温度和环空温度...")

        # 周围温度计算 - 修正：使用**而不是^
        R_cem = self.D_cem / 2  # 水泥环半径
        Tf = -0.0702 * R_cem ** (self.n_p) - 786.0309 * R_cem ** (self.n_s) + 1 / self.β_fx + 0.2927

        # 入流温度计算
        R_yg = self.D_youinner / 2  # 环空半径
        TI = -0.0702 * R_yg ** (self.n_p) - 786.0309 * R_yg ** (self.n_s) + 1 / self.β_fx + 0.2979

        # 环空温度
        R_hk = (self.D_yououter + self.D_taoinner) / 4
        Ta = -0.0702 * R_hk ** (self.n_p) - 786.0309 * R_hk ** (self.n_s) + 1 / self.β_fx

        # 保存参数
        self.Tf = Tf
        self.TI = TI
        self.Ta = Ta

        return Tf, TI, Ta

    def heattransfercoefficient(self):
        """油管壁到井筒水泥环的总传热系数、井筒水泥环到地层的总传热系数和油管壁到地层的总传热系数"""
        print("计算传热系数...")

        # 油管壁到井筒水泥环的总传热系数
        R_to = self.D_yououter / 2  # 油管外半径
        R_ti = self.D_youinner / 2  # 油管内半径
        R_co = self.D_taoouter / 2  # 套管外半径
        R_ci = self.D_taoinner / 2  # 套管内半径
        R_cem = self.D_cem / 2  # 水泥环半径

        K_t = self.K_you  # 油管热传导系数
        K_cas = self.K_tao  # 套管热传导系数
        K_cem = self.K_cem  # 水泥环导热系数

        a5 = R_to / K_t * np.log(R_to / R_ti)
        a6 = R_to / K_cas * np.log(R_co / R_ci)
        a7 = R_to / K_cem * np.log(R_cem / R_co)
        U_twb = 1 / (a5 + a6 + a7)

        # 井筒水泥环到地层的总传热系数
        Te = self.T_reservoir_initial  # 油藏顶层温度
        U_wbf = R_to * U_twb * (self.Ta - self.TI) / (R_cem * (self.TI - Te))

        # 油管壁到地层的总传热系数
        U_tf = 1 / (1 / U_twb + R_to / (R_cem * U_wbf))

        # 保存传热系数
        self.U_twb = U_twb
        self.U_wbf = U_wbf
        self.U_tf = U_tf
        self.R_to = R_to
        self.R_ti = R_ti
        self.R_co = R_co
        self.R_ci = R_ci
        self.R_cem = R_cem

        return U_twb, U_wbf, U_tf

    def generate_grids(self):
        """生成计算网格"""
        print("生成计算网格...")

        # 井筒网格
        self.nx_well = 100
        self.dx_well = self.l_well / (self.nx_well - 1)
        self.x_well = np.linspace(0, self.l_well, self.nx_well)

        # 油藏网格
        self.nx_res = 50
        self.ny_res = 25
        self.nz_res = 10
        self.x_res = np.linspace(0, self.l_reservoir, self.nx_res)
        self.y_res = np.linspace(-self.w_reservoir / 2, self.w_reservoir / 2, self.ny_res)
        self.z_res = np.linspace(0, self.h_reservoir, self.nz_res)

        # 创建网格坐标
        self.X_res, self.Y_res, self.Z_res = np.meshgrid(self.x_res, self.y_res, self.z_res, indexing='ij')

    def friction_factor(self, Re_m, ε_D):
        """计算摩擦系数"""
        if Re_m <= 2300:
            return 64 / Re_m
        elif Re_m >= 4000:
            return 1 / (1.8 * np.log(69.9 / Re_m + (ε_D / 3.7) ** 1.11)) ** 2
        else:  # 2300 < Re_m < 4000
            return 0.3164 / (Re_m) ** 0.25

    def calculate_wellbore_pressure(self):
        """计算井筒压力分布"""
        print("计算井筒压力分布...")  # 修正打印信息

        # 初始化压力数组
        P_well = np.zeros(self.nx_well)
        P_well[0] = self.P_heel  # 趾端压力边界条件

        # 初始化流量数组
        Q_well = np.zeros(self.nx_well)

        # 从趾端到跟端计算累计流量
        cumulative_flow = 0
        for i in range(0, self.nx_well):
            x_pos = self.x_well[i]

            # 检查当前位置是否有裂缝流入
            for j, frac_pos in enumerate(self.pos_fracture):
                if abs(x_pos - frac_pos) < self.dx_well / 2:
                    cumulative_flow += self.Q1_fractures[j]
                    break

            Q_well[i] = cumulative_flow


        # 计算压力分布 (从跟端到趾端)
        for i in range(1, self.nx_well):
            # 当前段的平均流量
            Q_avg = (Q_well[i] + Q_well[i - 1]) / 2

            if Q_avg > 0:
                # 流速
                velocity = Q_avg / (np.pi * (self.D_youinner / 2) ** 2)

                # 计算相对粗糙度
                ε_D = self.E_you / self.D_youinner

                # 雷诺数 - 修正：使用正确的粘度
                Re_m = self.ρ_f * velocity * self.D_youinner / self.μ_fx  # 去掉1e-3，因为μ_fx已经是Pa·s

                # 摩擦系数
                f = self.friction_factor(Re_m, ε_D)

                # 摩擦压降
                deltaP_friction = (f * self.dx_well * self.ρ_f * velocity ** 2 / (2 * self.D_youinner))

                # 重力压降
                deltaP_accel = self.ρ_f * self.dx_well * self.g * np.sin(np.pi * self.θ / 180)

                # 总压降
                deltaP_total = deltaP_friction + deltaP_accel

            else:
                deltaP_total = 0

            # 检查当前位置是否有裂缝流入
            x_pos = self.x_well[i]
            for j, frac_pos in enumerate(self.pos_fracture):
                if abs(x_pos - frac_pos) < self.dx_well / 2:
                    # 简化混合压降模型
                    S = np.pi * (self.D_youinner / 2) ** 2
                    Q = self.ρ_f * self.Q1_fractures[j] * (2 * Q_well[i - 1] - self.Q1_fractures[j])
                    deltaP_mixing = 3 * Q / S ** 2 / 2
                    deltaP_total += deltaP_mixing
                    break

            P_well[i] = P_well[i - 1] - deltaP_total

        self.P_well = P_well
        self.Q_well = Q_well
        return P_well

    def calculate_reservoir_pressure(self):
        """计算油藏压力分布"""
        print("计算油藏压力分布...")

        # 初始化压力场
        P_reservoir = np.ones_like(self.X_res) * self.P_reservoir_initial

        # 获取裂缝压力
        fracture_pressures = []
        for pos in self.pos_fracture:
            idx = np.argmin(np.abs(self.x_well - pos))
            fracture_pressures.append(self.P_well[idx])

        # 计算每条裂缝周围的压力分布
        for i, (x_frac, p_frac) in enumerate(zip(self.pos_fracture, fracture_pressures)):
            y_frac = 0  # 假设裂缝在y=0位置

            for ix in range(self.nx_res):
                for iy in range(self.ny_res):
                    for iz in range(self.nz_res):
                        # 计算到裂缝的距离
                        dx = abs(self.x_res[ix] - x_frac)
                        dy = abs(self.y_res[iy] - y_frac)
                        distance = np.sqrt(dx ** 2 + dy ** 2)

                        if distance > 0:
                            # 稳态径向流压力分布
                            r_e = self.h_fractures[i]  # 泄油半径, m
                            r_w = self.w_fractures[i] / 2  # 等效井筒半径

                            pressure_drop = (self.Q1_fractures[i] * self.μ_fx /  # 修正：使用self.μ_fx
                                             (2 * np.pi * self.k_fractures * self.h_reservoir) *
                                             np.log(r_e / max(distance, r_w)))

                            # 更新压力场
                            new_pressure = p_frac + pressure_drop
                            if new_pressure < P_reservoir[ix, iy, iz]:
                                P_reservoir[ix, iy, iz] = new_pressure

        self.P_reservoir = P_reservoir
        return P_reservoir

    def calculate_temperature(self):
        """计算温度分布"""
        print("计算温度分布...")

        # 井筒温度分布
        T_well = np.ones(self.nx_well) * self.T_reservoir_initial

        for i in range(1, self.nx_well):
            # 传热温度损失
            Q_avg = (self.Q_well[i] + self.Q_well[i - 1]) / 2  # 修正：使用self.Q_well

            if Q_avg > 0:
                # 传热损失
                a8 = self.Cp_fx * self.ρ1 * Q_avg
                deltaT_C = 2 * np.pi * self.R_ti * self.U_tf * (self.Tf - T_well[i - 1]) / a8  # 修正：使用T_well

                # 重力温度损失
                deltaT_G = self.g * np.sin(np.pi * self.θ / 180) / self.Cp_fx

                # 总损失
                deltaT_total = deltaT_C + deltaT_G
            else:
                deltaT_total = 0

            # 检查当前位置是否有裂缝流入
            x_pos = self.x_well[i]
            for j, frac_pos in enumerate(self.pos_fracture):
                if abs(x_pos - frac_pos) < self.dx_well / 2:
                    # 温度模型
                    K_JTo = (self.β_o * T_well[i - 1] - 1) / (self.ρ_o * self.Cp_o)  # 修正：使用T_well
                    K_JTw = (self.β_w * T_well[i - 1] - 1) / (self.ρ_w * self.Cp_w)
                    K_JTg = (self.β_g * T_well[i - 1] - 1) / (self.ρ_g * self.Cp_g)
                    K_JTm = (self.x_o * K_JTo * self.ρ_o + self.x_w * K_JTw * self.ρ_w + self.x_g * K_JTg * self.ρ_g) / self.ρ_f
                    deltaP = self.P_well[i] - self.P_well[i - 1]
                    deltaT_JT = K_JTm * deltaP

                    a9 = -self.η * 2 * np.pi * self.R_ti * self.U_tf * (self.Tf - T_well[i - 1])  # 修正：使用self.Tf
                    a10 = self.Cp_fx * self.ρ1 * self.Q1_fractures[j]  # 修正：使用当前裂缝的流量
                    a11 = a10 * (self.TI - T_well[i - 1])  # 修正：使用self.TI
                    deltaT_mixing = abs((a9 + a11) / a8 + deltaT_JT)
                    deltaT_total += deltaT_mixing
                    break

            T_well[i] = T_well[i - 1] + deltaT_total

        # 油藏温度分布
        T_reservoir = np.ones_like(self.X_res) * self.T_reservoir_initial

        # 更新裂缝附近的温度
        for i, x_frac in enumerate(self.pos_fracture):
            # 找到裂缝对应的井筒温度
            idx_well = np.argmin(np.abs(self.x_well - x_frac))
            T_frac = T_well[idx_well]

            for ix in range(self.nx_res):
                for iy in range(self.ny_res):
                    for iz in range(self.nz_res):
                        dx = abs(self.x_res[ix] - x_frac)
                        if dx < self.l_fractures[i]:
                            # 裂缝附近温度受井筒温度影响
                            T_reservoir[ix, iy, iz] = T_frac

        self.T_well = T_well
        self.T_reservoir = T_reservoir
        return T_well, T_reservoir

    def run_simulation(self):  # 修正：正确的缩进级别
        """运行完整模拟"""
        print("开始运行裂缝与水平井耦合模型...")

        # 按正确顺序调用方法
        self.preprocess_data()
        self.parameter()
        self.Inlettemperature()
        self.heattransfercoefficient()

        self.generate_grids()
        self.calculate_wellbore_pressure()
        self.calculate_reservoir_pressure()
        self.calculate_temperature()

        print("模拟完成!")

    def plot_results(self):  # 修正：正确的缩进级别
        """绘制结果图表"""
        print("生成结果图表……")

        fig = plt.figure(figsize=(20, 12))

        # 1. 井筒压力分布
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(self.x_well, self.P_well / 1e6, 'b-', linewidth=2)
        for pos in self.pos_fracture:
            plt.axvline(x=pos, color='red', linestyle='--', alpha=0.7)
            idx = self.pos_fracture.index(pos)
            plt.text(pos, np.max(self.P_well / 1e6) * 0.95,
                     f'Frac{idx + 1}', ha='center', va='bottom')
        plt.xlabel('水平井位置 (m)')
        plt.ylabel('压力 (MPa)')
        plt.title('水平井压力分布')
        plt.grid(True, alpha=0.3)

        # 2. 井筒温度分布
        ax2 = plt.subplot(2, 3, 2)
        plt.plot(self.x_well, self.T_well, 'r-', linewidth=2)
        for pos in self.pos_fracture:
            plt.axvline(x=pos, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('水平井位置 (m)')
        plt.ylabel('温度 (°C)')
        plt.title('水平井温度分布')
        plt.grid(True, alpha=0.3)

        # 3. 流量分布
        ax3 = plt.subplot(2, 3, 3)
        plt.plot(self.x_well, self.Q_well * 86400, 'g-', linewidth=2)  # 转换为m³/d
        for pos in self.pos_fracture:
            plt.axvline(x=pos, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('水平井位置 (m)')
        plt.ylabel('累计流量 (m³/d)')
        plt.title('井筒累计流量分布')
        plt.grid(True, alpha=0.3)

        # 4. 油藏压力剖面
        ax4 = plt.subplot(2, 3, 4)
        if hasattr(self, 'P_reservoir'):
            im = plt.contourf(self.X_res[:, :, 0], self.Y_res[:, :, 0],
                             self.P_reservoir[:, :, 0] / 1e6, 20, cmap='viridis')
            plt.colorbar(im, label='压力 (MPa)')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('油藏压力分布 (z=0)')

        # 5. 油藏温度剖面
        ax5 = plt.subplot(2, 3, 5)
        if hasattr(self, 'T_reservoir'):
            im = plt.contourf(self.X_res[:, :, 0], self.Y_res[:, :, 0],
                             self.T_reservoir[:, :, 0], 20, cmap='hot')
            plt.colorbar(im, label='温度 (°C)')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('油藏温度分布 (z=0)')

        plt.tight_layout()
        plt.show()

    def export_data_to_excel(self, filename="水平井模拟结果.xlsx"):
        """导出所有数据到Excel文件"""
        print(f"导出数据到Excel文件: {filename}")

        try:
            # 创建Excel写入器
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:

                # 1. 导出井筒数据
                well_data = pd.DataFrame({
                    '位置_m': self.x_well,
                    '压力_Pa': self.P_well,
                    '压力_MPa': self.P_well / 1e6,
                    '温度_C': self.T_well,
                    '流量_m3s': self.Q_well,
                    '流量_m3d': self.Q_well * 86400
                })
                well_data.to_excel(writer, sheet_name='井筒数据', index=False)

                # 2. 导出裂缝数据
                fracture_data = pd.DataFrame({
                    '裂缝编号': [f'Frac{i + 1}' for i in range(self.N_fractures)],
                    '位置_m': self.pos_fracture,
                    '半长_m': self.l_fractures,
                    '宽度_m': self.w_fractures,
                    '高度_m': self.h_fractures,
                    '油相产量_m3d': self.Qo_fractures,
                    '水相产量_m3d': self.Qw_fractures,
                    '气相产量_m3d': self.Qg_fractures,
                    '总产量_m3d': [qo + qw + qg for qo, qw, qg in
                                zip(self.Qo_fractures, self.Qw_fractures, self.Qg_fractures)]
                })
                fracture_data.to_excel(writer, sheet_name='裂缝数据', index=False)

                # 3. 导出油藏数据 (简化版本，只导出第一层)
                if hasattr(self, 'P_reservoir') and hasattr(self, 'T_reservoir'):
                    # 压力数据
                    pressure_2d = self.P_reservoir[:, :, 0] / 1e6  # 转换为MPa
                    pressure_df = pd.DataFrame(pressure_2d,
                                               index=[f'Y_{i}' for i in range(pressure_2d.shape[0])],
                                               columns=[f'X_{i}' for i in range(pressure_2d.shape[1])])
                    pressure_df.to_excel(writer, sheet_name='油藏压力_MPa')

                    # 温度数据
                    temp_2d = self.T_reservoir[:, :, 0]
                    temp_df = pd.DataFrame(temp_2d,
                                           index=[f'Y_{i}' for i in range(temp_2d.shape[0])],
                                           columns=[f'X_{i}' for i in range(temp_2d.shape[1])])
                    temp_df.to_excel(writer, sheet_name='油藏温度_C')

                # 4. 导出模型参数
                params_data = {
                    '参数名称': [
                        '水平段长度_m', '水泥环半径_m', '套管外径_m', '套管内径_m',
                        '油管外径_m', '油管内径_m', '初始压力_Pa', '初始温度_C',
                        '油藏长度_m', '油藏宽度_m', '油藏高度_m', '孔隙度',
                        '油相密度_kg_m3', '水相密度_kg_m3', '气相密度_kg_m3'
                    ],
                    '参数值': [
                        self.l_well, self.D_cem, self.D_taoouter, self.D_taoinner,
                        self.D_yououter, self.D_youinner, self.P_reservoir_initial, self.T_reservoir_initial,
                        self.l_reservoir, self.w_reservoir, self.h_reservoir, self.Φ_reservoir,
                        self.ρ_o, self.ρ_w, self.ρ_g
                    ]
                }
                params_df = pd.DataFrame(params_data)
                params_df.to_excel(writer, sheet_name='模型参数', index=False)

                # 5. 导出计算结果汇总
                summary_data = {
                    '计算项目': [
                        '总流量_m3s', '混合流体粘度_Pas', '混合流体比热容_J_kgK',
                        '油相体积分数', '水相体积分数', '气相体积分数',
                        '传热系数_U_twb', '传热系数_U_wbf', '传热系数_U_tf'
                    ],
                    '计算结果': [
                        self.Q_t, self.μ_fx, self.Cp_fx,
                        self.x_o, self.x_w, self.x_g,
                        self.U_twb, self.U_wbf, self.U_tf
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='计算结果汇总', index=False)

            print(f"数据成功导出到 {filename}")
            return True

        except Exception as e:
            print(f"导出Excel文件时出错: {e}")
            return False

    def export_data_to_csv(self, folder="output_data"):
        """导出数据到多个CSV文件"""
        print(f"导出数据到CSV文件，文件夹: {folder}")

        try:
            # 创建输出文件夹
            if not os.path.exists(folder):
                os.makedirs(folder)

            # 1. 井筒数据
            well_df = pd.DataFrame({
                '位置_m': self.x_well,
                '压力_Pa': self.P_well,
                '压力_MPa': self.P_well / 1e6,
                '温度_C': self.T_well,
                '流量_m3s': self.Q_well,
                '流量_m3d': self.Q_well * 86400
            })
            well_df.to_csv(f"{folder}/well_data.csv", index=False)

            # 2. 裂缝数据
            fracture_df = pd.DataFrame({
                '裂缝编号': [f'Frac{i + 1}' for i in range(self.N_fractures)],
                '位置_m': self.pos_fracture,
                '半长_m': self.l_fractures,
                '宽度_m': self.w_fractures,
                '高度_m': self.h_fractures,
                '导流能力_mDm': self.c_fractures,
                '油相产量_m3d': self.Qo_fractures,
                '水相产量_m3d': self.Qw_fractures,
                '气相产量_m3d': self.Qg_fractures
            })
            fracture_df.to_csv(f"{folder}/fracture_data.csv", index=False)

            # 3. 油藏数据
            if hasattr(self, 'P_reservoir') and hasattr(self, 'T_reservoir'):
                # 压力数据
                np.savetxt(f"{folder}/reservoir_pressure_MPa.csv",
                           self.P_reservoir[:, :, 0] / 1e6, delimiter=",")
                # 温度数据
                np.savetxt(f"{folder}/reservoir_temperature_C.csv",
                           self.T_reservoir[:, :, 0], delimiter=",")

                # 网格坐标
                np.savetxt(f"{folder}/x_coordinates.csv", self.x_res, delimiter=",")
                np.savetxt(f"{folder}/y_coordinates.csv", self.y_res, delimiter=",")

            # 4. 参数文件
            with open(f"{folder}/simulation_parameters.txt", "w") as f:
                f.write("水平井模拟参数汇总\n")
                f.write("=" * 50 + "\n")
                f.write(f"水平段长度: {self.l_well} m\n")
                f.write(f"初始油藏压力: {self.P_reservoir_initial / 1e6:.2f} MPa\n")
                f.write(f"初始油藏温度: {self.T_reservoir_initial} °C\n")
                f.write(f"总流量: {self.Q_t:.6f} m³/s\n")
                f.write(f"混合流体粘度: {self.μ_fx:.6f} Pa·s\n")
                f.write(f"混合流体比热容: {self.Cp_fx:.2f} J/kg·K\n")
                f.write(f"油相体积分数: {self.x_o:.3f}\n")
                f.write(f"水相体积分数: {self.x_w:.3f}\n")
                f.write(f"气相体积分数: {self.x_g:.3f}\n")

            print(f"CSV文件导出完成，保存在 {folder} 文件夹")
            return True

        except Exception as e:
            print(f"导出CSV文件时出错: {e}")
            return False

    def export_summary_report(self, filename="模拟结果摘要.txt"):
        """生成模拟结果摘要报告"""
        print(f"生成摘要报告: {filename}")

        try:
            with open(filename, "w", encoding='utf-8') as f:
                f.write("水平井温度压力模拟结果摘要\n")
                f.write("=" * 60 + "\n\n")

                f.write("1. 基本参数\n")
                f.write("-" * 30 + "\n")
                f.write(f"水平段长度: {self.l_well} m\n")
                f.write(f"裂缝数量: {self.N_fractures} 条\n")
                f.write(f"初始油藏压力: {self.P_reservoir_initial / 1e6:.2f} MPa\n")
                f.write(f"初始油藏温度: {self.T_reservoir_initial} °C\n\n")

                f.write("2. 流量信息\n")
                f.write("-" * 30 + "\n")
                f.write(f"总流量: {self.Q_t * 86400:.2f} m³/d\n")
                f.write(f"油相流量: {sum(self.Qo_fractures):.2f} m³/d\n")
                f.write(f"水相流量: {sum(self.Qw_fractures):.2f} m³/d\n")
                f.write(f"气相流量: {sum(self.Qg_fractures):.2f} m³/d\n\n")

                f.write("3. 井筒结果统计\n")
                f.write("-" * 30 + "\n")
                f.write(f"井筒压力范围: {np.min(self.P_well) / 1e6:.2f} - {np.max(self.P_well) / 1e6:.2f} MPa\n")
                f.write(f"井筒温度范围: {np.min(self.T_well):.2f} - {np.max(self.T_well):.2f} °C\n")
                f.write(f"压力降: {(self.P_well[0] - self.P_well[-1]) / 1e6:.2f} MPa\n")
                f.write(f"温度变化: {self.T_well[0] - self.T_well[-1]:.2f} °C\n\n")

                f.write("4. 裂缝信息\n")
                f.write("-" * 30 + "\n")
                for i in range(self.N_fractures):
                    f.write(f"裂缝 {i + 1}: 位置 {self.pos_fracture[i]} m, "
                            f"产量 {self.Qo_fractures[i] + self.Qw_fractures[i] + self.Qg_fractures[i]:.1f} m³/d\n")

                f.write("\n5. 关键计算结果\n")
                f.write("-" * 30 + "\n")
                f.write(f"混合流体粘度: {self.μ_fx:.6f} Pa·s\n")
                f.write(f"混合流体比热容: {self.Cp_fx:.0f} J/kg·K\n")
                f.write(f"传热系数 U_tf: {self.U_tf:.4f} W/m²·K\n")
                f.write(f"体积分数 - 油: {self.x_o:.3f}, 水: {self.x_w:.3f}, 气: {self.x_g:.3f}\n")

            print(f"摘要报告已保存到 {filename}")
            return True

        except Exception as e:
            print(f"生成摘要报告时出错: {e}")
            return False

    def export_all_data(self, base_name="水平井模拟"):
        """导出所有数据格式"""
        print("开始导出所有数据...")

        # Excel文件
        excel_file = f"{base_name}.xlsx"
        self.export_data_to_excel(excel_file)

        # CSV文件
        csv_folder = f"{base_name}_CSV数据"
        self.export_data_to_csv(csv_folder)

        # 摘要报告
        report_file = f"{base_name}_摘要报告.txt"
        self.export_summary_report(report_file)

        print("所有数据导出完成！")
        print(f"- Excel文件: {excel_file}")
        print(f"- CSV文件夹: {csv_folder}")
        print(f"- 摘要报告: {report_file}")


# 使用示例
if __name__ == "__main__":
    # 创建模型实例
    model = HorizontalWellTemperature()

    # 运行模拟
    model.run_simulation()

    # 绘制图表
    model.plot_results()

    # 导出数据
    model.export_all_data("我的水平井模拟")
