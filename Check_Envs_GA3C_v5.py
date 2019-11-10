# NEW COLORS 108.04.24
# output=gray colors
import numpy as np
import pygame
import time

# Define some colors
COLORS = 3  # 測試次數上限

# 模擬器上顏色設定
BLACK = np.array((0, 0, 0))
WHITE = np.array((255, 255, 255))
BLUE = np.array((60, 150, 255))
PURPLE = np.array((153, 47, 185))
RED_PROBE = np.array((230, 90, 80))
YELLOW = np.array((235, 226, 80))

# 輸出圖顏色設定
BACKGROUND_COLORS = 255  # 背景
BUFFER_COLORS = 170  # 緩衝區
PROBE_COLORS = 220  # 探針
# 其他測試次數狀態
OTHER_COLORS = 129
NUM_COLORS = [] # ex: 測試上限3次 [129, 86, 43]
for num in range(COLORS):
    NUM_COLORS.append(int(OTHER_COLORS * (1 - num / COLORS)))

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 1   # 實際環境圖,一像素代表一晶粒
HEIGHT = 1  # 實際環境圖
WIDTH_sc = 20  # 模擬器顯示畫面
HEIGHT_sc = 20  # 模擬器顯示畫面

# This sets the margin between each cell
MARGIN = 0     # 實際環境圖
MARGIN_sc = 2  # 模擬器顯示畫面

# Probe's location when the environment initialize
Initial = [(2, 2), (14, 14), (2, 14), (14, 2), (11, 5), (5, 11), (11, 11), (5, 5), (8, 8)]
PACE = 1  # 移動步伐

class wafer_check():
    def __init__(self,wafer,probe,mode=0,training_time=60,training_steps=0):
        self._envs = np.array(wafer)  # 晶圓由 -1, 0表示(-1代表緩衝區, 0代表待測試晶粒）
        self._envs_nan = np.zeros(self._envs.shape)  # 晶圓由 nan, 0 表示(nan代表緩衝區, 0代表待測試晶粒）
        self._probe = np.array(probe, np.int)      # 探針卡由 0,1表示
        self.envsY, self.envsX = self._envs.shape  # 晶圓長寬
        self.wafer_len = self.envsY * self.envsX  # 晶粒總數
        self.probY, self.probX = self._probe.shape  # 探針長寬
        self.probZ = max(self.probY, self.probX)  # 探針最長邊
        self.envs_list = [(b,a) for b in range(self.envsY) for a in range(self.envsX) if self._envs[b,a] == -1]  # 緩衝區位置
        self.envs_len = len(self.envs_list)  # 緩衝區數量
        self.probe_list = [(b,a) for b in range(self.probY) for a in range(self.probX) if self._probe[b,a] == 1]  # 探針形狀
        self.probe_len = len(self.probe_list)  # 探針數量
        self.size = [(self.envsX*WIDTH+(self.envsX+1)*MARGIN),
                     (self.envsY*HEIGHT+(self.envsY+1)*MARGIN)]                         # 實際環境圖尺寸
        self.size_sc = [(self.envsX*WIDTH_sc+(self.envsX+1)*MARGIN_sc),
                      (self.envsY*HEIGHT_sc+(self.envsY+1)*MARGIN_sc)]                  # 模擬器顯示畫面尺寸
        self._output = np.full((self.size[1],self.size[0]), BACKGROUND_COLORS, np.int)  # 初始化輸出圖
        self.location = np.array(Initial)  # 初始位置
        self.action_space = ['None','Down','Right','Up','Left','Down-Right','Up-Right','Up-Left','Down-Left']
        self.action_space_num = int((len(self.action_space) - 1) * PACE)  # 行為總數(為8個方向 * 移動步伐）
        self.available = np.zeros(self.action_space_num, dtype=np.float32)  # 表示可移動行為之向量
        self.num_max = COLORS
        self.reward_value = 0  # 獎勵
        self.envs_mean = None  # 所有晶粒被測試過次數平均
        self.envs_std = None   # 所有晶粒被測試過次數標準差
        self.mode = mode       # 是否顯示模擬畫面(是 = 1 ,否= 0)

        #  限制一回合最長可訓練時間(若設小於0則訓練時間為無限制）
        if training_time > 0:
            self.training_time = training_time
        else:
            self.training_time = np.inf

        #  限制一回合最多可移動步數(若設小於0則移動步數為無限制）
        if training_steps > 0:
            self.training_steps = training_steps
        else:
            self.training_steps = np.inf

        # 是否顯示模擬畫面(是 = 1 ,否= 0)
        if self.mode == 1:
            self.sc = pygame.display.set_mode(self.size_sc)

        # 初始化輸出圖
        self.reset_observation()
        # 初始化環境
        self.reset()

    # 計算方形尺寸
    @staticmethod
    def rect(column, row):
        rect = [(MARGIN_sc + WIDTH_sc) * column + MARGIN_sc,
                (MARGIN_sc + HEIGHT_sc) * row + MARGIN_sc,
                WIDTH_sc,
                HEIGHT_sc]
        return rect

    # 於圖output上填顏色
    @staticmethod
    def draw_plt(output, y, x, color):  # X : column, Y : row
        for h in range(HEIGHT):
            for w in range(WIDTH):
                output_h = y * HEIGHT + h
                output_w = x * WIDTH + w
                output[output_h][output_w] = color

    def reset(self):
        #reset the environment
        self.y, self.x = self.location[np.random.randint(len(self.location))]  # 隨機取一個初始位置為y, x
        self.y_last, self.x_last = self.y, self.x
        self.steps = 0  # 移動步署
        self.dist = 0  # 移動距離
        self.num_color = np.zeros(self.num_max+2, np.int)  # 表示各個晶粒狀態的個數[未測試過, 已測試1次, 已測試2次, 已測試3次以上, 緩衝區]
        self.action = 'None'
        self.reward_value = 0
        self.envs = np.copy(self._envs_nan)  # 重新拷貝初始晶圓狀態
        self.output = np.copy(self._output)  # 重新拷貝初始輸出圖

        if self.mode == 1:    # 若有模擬畫面，畫面也須初始化
            self.reset_envs()

        # 將初始探針位置的晶圓狀態改為測試一次
        for b in range(self.probY):
            for a in range(self.probX):
                if self._probe[b][a] == 1 and not np.isnan(self.envs[self.y+b][self.x+a]):
                    self.envs[self.y+b][self.x+a] = 1
        self.num_color_last = np.zeros(self.num_max+2, np.int)  # 表示前一次移動之各個晶粒狀態的個數
        self.num_color_last[-1] = self.envs_len  # 緩衝區個數
        self.num_color_last[0] = (self._envs == 0).sum()  # 未測試過數
        self.time_end = time.time() + self.training_time  # 有時間限制，最終訓練時刻
        self.step()
        return self.output, self.available

    def step(self, action=None):
        #Agent's action
        now = time.time()

        if action != None:
            act = ((action) % 8)  # 動作選擇(0~7)
            pace = int((action) / 8) + 1  # 動作移動步伐

        self.done = 0  # 測試終止為1
        self.envs_mean = None
        self.envs_std = None
        self.time_is_end = 0   # 時間限制，測試終止
        self.steps_is_end = 0  # 總步數限制，測試終止
        self.episode_is_end = 0  # 所有晶粒皆已測試完成，測試終止
        self.reward_value = 0

        if now < self.time_end and self.steps < self.training_steps:

            y = self.y
            x = self.x
            y_diff = self.envsY-self.probY  # 探針座標於 y 方向最低位置
            x_diff = self.envsX-self.probX  # 探針座標於 x 方向最低位置
            print(y_diff, x_diff)

            probe_list = self.probe_list

            invalid = 0
            self.steps += 1  # 移動步數累計加1

            # move the probe
            if action == None:  # 若為Ｎone則移動步數修正，減1
                invalid = -1
                self.steps -= 1
                self.action = 'None'
            elif pace > self.probZ:  # 若步伐大於探針尺寸，視為無效行動
                invalid = -1
                self.steps -= 1
                self.action = 'None'
            elif act == 0:
                if (y+pace-1) < y_diff:
                    y += pace
                    invalid = 0
                    self.action = 'Down'
                else:
                    invalid = 1
            elif act == 1:
                if (x+pace-1) < x_diff:
                    x += pace
                    invalid = 0
                    self.action = 'Right'
                else:
                    invalid = 1
            elif act == 2:
                if (y-pace+1) > 0:
                    y -= pace
                    invalid = 0
                    self.action = 'Up'
                else:
                    invalid = 1
            elif act == 3:
                if (x - pace+1) > 0:
                    x -= pace
                    invalid = 0
                    self.action = 'Left'
                else:
                    invalid = 1
            elif act == 4:
                if (y+pace-1) < y_diff and (x+pace-1) < x_diff:
                    y += pace
                    x += pace
                    invalid = 0
                    self.action = 'Down-Right'
                else:
                    invalid = 1
            elif act == 5:
                if (y-pace+1) > 0 and (x+pace-1) < x_diff:
                    y-=pace
                    x+=pace
                    invalid = 0
                    self.action = 'Up-Right'
                else:
                    invalid = 1
            elif act == 6:
                if (y-pace+1) > 0 and (x-pace+1) > 0:
                    y-=pace
                    x-=pace
                    invalid = 0
                    self.action = 'Up-Left'
                else:
                    invalid = 1
            elif act == 7:
                if (y+pace-1) < y_diff and (x-pace+1) > 0:
                    y+=pace
                    x-=pace
                    invalid = 0
                    self.action = 'Down-Left'
                else:
                    invalid = 1
            else:
                invalid = -1
                self.action = 'None'

            # 無效動作
            if invalid == 1:
                self.action = 'Invalid'
            # 有效動作
            elif invalid == 0:
                # 更新探針座標位置
                self.y = y
                self.x = x
                # 探針位置的晶圓測試狀態累加一次
                for c in range(len(probe_list)):
                    self.envs[y+probe_list[c][0]][x+probe_list[c][1]] += 1

        elif now >= self.time_end:
            self.time_is_end = 1

        if self.steps >= self.training_steps:
            self.steps_is_end = 1

        self.check()  # 統計晶粒狀態並計算獎勵

        self.observation()

        self.action_available()

        if self.mode == 1:
            self.build_envs()
            time.sleep(0.01)

        self.y_last = self.y
        self.x_last = self.x

        if self.steps_is_end == 1:
            self.steps = 0

        if self.time_is_end == 1:
            self.steps = 0
            self.time_end = time.time() + self.training_time

        return self.output, self.reward_value, self.done, self.available, self.envs_mean, self.envs_std
    
    def check(self):
        # 表示各個晶粒狀態的個數num_color[5] = [未測試過, 已測試1次, 已測試2次, 已測試3次以上, 緩衝區]
        self.num_color[-1] = self.envs_len  # 緩衝區數
        for n in range(0, self.num_max):
            self.num_color[n] = (self.envs == n).sum()

        self.num_color[-2] = self.wafer_len - sum(self.num_color) + self.num_color[-2]  # 已測試num_max次以上

        self.dist = np.sqrt(np.square(self.y - self.y_last)+np.square(self.x - self.x_last))  # 計算探針移動距離

        #calculate the reward
        if self.action != "None":

            #1st reward
            if self.num_color_last[0] - self.num_color[0] > 0:
                self.reward_value+=((self.num_color_last[0] - self.num_color[0])*0.01)
                if self.num_color_last[0] - self.num_color[0] == self.probe_len:
                    self.reward_value+=((self.num_color_last[0] - self.num_color[0])*0.01)

            #2nd reward
            for num in range(2,self.num_max+1):
                if self.num_color[num] - self.num_color_last[num] > 0:
                    self.reward_value-=(((self.num_color[num] - self.num_color_last[num])*num)*0.003)

            #3rd reward
            if np.array_equal(self.num_color,self.num_color_last):
                self.reward_value-=0.1

            #4th reward
            self.reward_value-=self.dist*0.01

        # 若測試終止
        if self.num_color[0] == 0 or self.time_is_end == 1 or self.steps_is_end == 1:
            self.envs_mean = np.nanmean(self.envs)  # 計算平均
            self.envs_std = np.nanstd(self.envs)  # 計算標準差
            
            #Stop the screen when the episode is end.
            if self.mode == 1:
                self.build_envs()   # 初始化模擬畫面
                time.sleep(0.1)

            #Initialize the environment
            self.action = 'None'
            self.done = 1   # 代表測試終止
            self.y, self.x = self.location[np.random.randint(len(self.location))]
            self.y_last, self.x_last = self.y, self.x
            self.dist = 0
            self.num_color = np.zeros(self.num_max+2,np.int)
            self.envs = np.copy(self._envs_nan)
            self.output = np.copy(self._output)
            if self.mode == 1:
                self.reset_envs()

            # 將初始探針位置的晶圓狀態改為測試一次
            for b in range(self.probY):
                for a in range(self.probX):
                    if self._probe[b][a] == 1 and not np.isnan(self.envs[self.y + b][self.x + a]):
                        self.envs[self.y + b][self.x + a] = 1

            self.envs_show = np.copy(self.envs)
            self.num_color[-1] = self.envs_len
            self.num_color[0] = (self.envs == 0).sum()
            self.num_color[1] = (self.envs == 1).sum()

            if self.time_is_end != 1 and self.steps_is_end != 1:
                # 代表成功完成所有晶粒測試
                self.episode_is_end = 1
                self.steps = 0

                #5th reward
                self.reward_value += 1

        self.num_color_last = np.copy(self.num_color)

    def observation(self):
        # 更新輸出圖

        probe_list = self.probe_list
        probe_len = self.probe_len

        # 畫探針走過位置的晶粒狀態
        for c in range(probe_len):
            for num in range(1, self.num_max+1):
                if self.envs[self.y_last+probe_list[c][0]][self.x_last+probe_list[c][1]] == num:  # 測試過1~3次
                    color = NUM_COLORS[num-1]
            if self.envs[self.y_last+probe_list[c][0]][self.x_last+probe_list[c][1]] > self.num_max:  # 測試過3次以上
                color = NUM_COLORS[self.num_max-1]
            if np.isnan(self.envs[self.y_last+probe_list[c][0]][self.x_last+probe_list[c][1]]):  # 緩衝區
                color = BUFFER_COLORS
            wafer_check.draw_plt(self.output, self.y_last + self.probe_list[c][0], self.x_last + self.probe_list[c][1], color)

        # 畫探針當下位置
        for c in range(probe_len):
            color = PROBE_COLORS
            wafer_check.draw_plt(self.output, self.y + self.probe_list[c][0], self.x + self.probe_list[c][1], color)


    def build_envs(self):
        # 更新模擬器顯示畫面

        # 畫探針走過位置的晶粒狀態
        for c in range(self.probe_len):
            if self.envs[self.y_last + self.probe_list[c][0]][self.x_last + self.probe_list[c][1]] >= 1:  # 走過一次以上
                color = (WHITE / self.num_max).astype(np.int)
            elif np.isnan(self.envs[self.y_last+self.probe_list[c][0]][self.x_last+self.probe_list[c][1]]): # 緩衝區
                color = YELLOW

            pygame.draw.rect(self.sc,
                             color,
                             wafer_check.rect((self.x_last + self.probe_list[c][1]),
                                              (self.y_last + self.probe_list[c][0])))

        # 畫探針當下位置
        for c in range(self.probe_len):
            color = RED_PROBE
            if self.action == 'Invalid':  # 若為無效動作，呈現紫色
                color = PURPLE
            pygame.draw.rect(self.sc,
                             color,
                             wafer_check.rect((self.x + self.probe_list[c][1]),
                                              (self.y + self.probe_list[c][0])))

        pygame.display.flip()

    def reset_observation(self):
        #  初始化輸出圖，繪製晶圓狀態
        color = BUFFER_COLORS
        for row in range(self.envsY):
            for column in range(self.envsX):
                if self._envs[row][column] == -1:
                    wafer_check.draw_plt(self._output, column, row, color)
                    self._envs_nan[row][column] = np.nan

    def reset_envs(self):
        # 初始化模擬器顯示畫面，繪製晶圓狀態

        self.sc.fill(BLACK)
        for row in range(self.envsY):
            for column in range(self.envsX):
                if self._envs[row][column] == -1:
                    pygame.draw.rect(self.sc, YELLOW, wafer_check.rect(row, column))  # 緩衝區
                else:
                    pygame.draw.rect(self.sc, BLUE, wafer_check.rect(row, column))  # 未測試區

    def action_available(self):
        # evaluate actions that will go beyond the boundary & produce vector to filter

        m = self.envsY
        n = self.envsX
        i = self.probY
        j = self.probX

        for k in range(self.action_space_num):
            
            act = k % 8
            step = k // 8 + 1
            
            y = self.y
            x = self.x
            
            if act == 0:
                if (y+step-1) < (m-i):
                    y+=step
                else:
                    self.available[k] = np.inf
                    continue
            elif act == 1:
                if (x+step-1) < (n-j):
                    x+=step
                else:
                    self.available[k] = np.inf
                    continue
            elif act == 2:
                if (y-step+1) > 0:
                    y-=step
                else:
                    self.available[k] = np.inf
                    continue
            elif act == 3:
                if (x-step+1) > 0:
                    x-=step
                else:
                    self.available[k] = np.inf
                    continue
            elif act == 4:
                if (y+step-1) < (m-i) and (x+step-1) < (n-j):
                    y+=step
                    x+=step
                else:
                    self.available[k] = np.inf
                    continue
            elif act == 5:
                if (y-step+1) > 0 and (x+step-1) < (n-j):
                    y-=step
                    x+=step
                else:
                    self.available[k] = np.inf
                    continue
            elif act == 6:
                if (y-step+1) > 0 and (x-step+1) > 0:
                    y-=step
                    x-=step
                else:
                    self.available[k] = np.inf
                    continue
            elif act == 7:
                if (y+step-1) < (m-i) and (x-step+1) > 0:
                    y+=step
                    x-=step
                else:
                    self.available[k] = np.inf
                    continue

            self.available[k] = 0

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    wafer = np.loadtxt('envs.txt')
    probe = np.loadtxt('probe.txt')

    envs = wafer_check(wafer, probe, mode=1, training_time=0, training_steps=1000)

    pygame.init()
    pygame.display.set_caption("Wafer Check Simulator")

    # Loop until the user clicks the close button.
    done = False

    while not done:

        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # 初始化環境
                    envs.reset()
                if event.key == pygame.K_s:
                    envs.step(0)
                if event.key == pygame.K_d:
                    envs.step(1)
                if event.key == pygame.K_w:
                    envs.step(2)
                if event.key == pygame.K_a:
                    envs.step(3)
                if event.key == pygame.K_c:
                    envs.step(4)
                if event.key == pygame.K_e:
                    envs.step(5)
                if event.key == pygame.K_q:
                    envs.step(6)
                if event.key == pygame.K_z:
                    envs.step(7)
                if event.key == pygame.K_p:  # 顯示輸出圖
                    plt.subplot(1, 2, 1), plt.title('rainbow')
                    plt.imshow(envs.output,cmap = 'rainbow')
                    plt.subplot(1, 2, 2), plt.title('gray')
                    plt.imshow(envs.output,cmap = 'gray')
                    plt.show()

    pygame.quit()
