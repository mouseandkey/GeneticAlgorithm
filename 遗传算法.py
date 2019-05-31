# coding=utf-8
import numpy as np
from PIL import Image
import copy
import random


# 适应度评价函数
def otsu(histogram: list, value: int) -> int:
    # 计算最大类间方差
    length_0, length_1, sum_0, sum_1 = 0, 0, 0, 0
    for i in range(256):
        if i <= value:
            length_0 += histogram[i]
            sum_0 += i * histogram[i]
        else:
            length_1 += histogram[i]
            sum_1 += i * histogram[i]
    # 防止被除数为0
    length_0 = length_0 if length_0 else length_0 + 1
    length_1 = length_1 if length_1 else length_1 + 1

    w0 = length_0 / sum(histogram)
    w1 = length_1 / sum(histogram)
    u0 = sum_0 / length_0
    u1 = sum_1 / length_1
    u = w0 * u0 + w1 * u1

    g = w0 * (u0 - u) * (u0 - u) + w1 * (u1 - u) * (u1 - u)  # 最大类间方差
    # g = w0*w1*(u0-u1)*(u0-u1)
    return g


class ThresholdSegmentationOfGrayImage:

    # 实例时，输入要处理的图片位置，并设置必要参数
    def __init__(self, histogram, figure, weight):
        self.histogram = histogram
        self.figure = figure  # 编码长度
        self.weight = weight  # 编码权重
        if len(weight) != self.figure:
            print("你输入的权重长度不合法")
            self.weight = [1 for i in range(self.figure)]
        else:
            for i in range(self.figure):
                if type(weight[i]) != int:
                    print('你输入的权重中不全是int类型\n已将weight设置为默认值，请使用set_weight方法修改')
                    self.weight = [1 for i in range(self.figure)]
                    break
        self.single_fitness = None  # 计算单个适应度的函数，由外部传入
        self.population = []  # 种群

        # 配置默认参数
        self.inti_config()

    # 编码函数（将十进制编码解码会二进制）
    def encode(self, value: int) -> list:
        # 没有用到 没写完
        #编二进制
        new_value = []
        for s in bin(value):
            if s == 'b':
                new_value = []
            else:
                new_value.append(s)
        #编权重

        return new_value

    #解码函数（将二进制解码成为十进制
    def decode(self,value:list)->int:
        #解权重
        self.weight
        weight_ness = []
        for i in range(len(self.weight)):
            weight_ness.append(value[sum(self.weight[:i]):sum(self.weight[:i+1])])
        value_wn = []
        for i in range(len(weight_ness)):
            if weight_ness[i].count('1') / len(weight_ness[i]) >= 0.5:
                value_wn.append('1')
            else:
                value_wn.append('0')
        #解二进制
        new_value = int(''.join(value_wn), 2)
        return new_value

    # s设置适应度函数
    def set_fitness(self, func):
        self.single_fitness = func

    # 默认的配置 用字典保存
    def inti_config(self):
        self.config = {
            'max_commutative_algebra': 1000,
            'population_size': 10,
            'exchange_rate': 0.8,
            'mutation_rate': 0.2
        }

    def get_config(self):
        return self.config

    def set_config(self, config):  # 默认参数
        for key in config.keys():
            if key in self.config.keys():
                self.config[key] = config[key]
            else:
                print("warning:可设置的参数中没有%s" % key)

    def init_population(self):  # 生成初始种群
        self.population = []
        for p in range(self.config['population_size']):
            individual = []
            for i in range(sum(self.weight)):  # 格式：['0','1','0','1','0',,,,]
                individual.append(random.choice('01'))
            self.population.append(individual)

    def fitness_compute(self):  # 计算所有个体适应度，并进行归一化操作
        # 计算所有个体适应度
        self.fitness = []
        for individual in self.population:
            self.fitness.append(self.single_fitness(self.histogram, self.decode(individual)))
        # 归一化操作
        self.fit = []
        fitness_sum = sum(self.fitness)
        for i in range(len(self.fitness)):
            self.fit.append(self.fitness[i] / fitness_sum)

    def selection(self):  # 选择
        # 轮盘赌选择
        fit_section = [sum(self.fit[:i+1]) for i in range(len(self.fit))]
        choice = np.random.uniform(0, 1, len(self.fit))
        choice.sort()
        fit_index = 0
        new_index = 0
        new_population = []
        while new_index < self.config['population_size']:
            # print(new_index,fit_index)
            if choice[new_index] < fit_section[fit_index]:
                new_population.append(copy.deepcopy(self.population[fit_index]))
                new_index += 1
            else:
                fit_index += 1
        self.population = new_population
        # 随机竞争选择


    def crossover(self):  # 交换
        # 单点交叉 多次交叉
        change = np.random.uniform(0, 1)
        if change < self.config['exchange_rate']:
            change_position = np.random.randint(0, int(sum(self.weight) * 0.8))
            change_lenght = np.random.randint(1, int(sum(self.weight) / 2))  # 可设置
            # individual_1 = np.random.randint(0,self.config['population_size'])
            # individual_2 = np.random.randint(0,self.config['population_size'])
            sort = [i for i in range(self.config['population_size'])]
            np.random.shuffle(sort)

            # while(individual_1 == individual_2):
            #   individual_2 = np.random.randint(0,self.config['population_size'])
            # if change_position+change_lenght >= self.figure:
            #   change_lenght = self.figure-change_position-1 # 可以越界，会自动判定
            # temp_change = self.population[individual_1][change_position:change_position+change_lenght]
            # self.population[individual_1] = self.population[individual_1][:change_position]+ \
            # self.population[individual_2][change_position:change_position+change_lenght]+ \
            # self.population[individual_1][change_position+change_lenght:]
            # self.population[individual_2] = self.population[individual_2][:change_position]+ \
            # temp_change+self.population[individual_2][change_position+change_lenght:]
            self.change_num = int(self.config['population_size'] / 2)  # 可设置调节
            for i in range(self.change_num):
                temp_change = self.population[i][change_position:change_position + change_lenght]
                self.population[i] = self.population[i][:change_position] + \
                                     self.population[i * 2][change_position:change_position + change_lenght] + \
                                     self.population[i][change_position + change_lenght:]
                self.population[i * 2] = self.population[i * 2][:change_position] + \
                                         temp_change + self.population[i * 2][change_position + change_lenght:]
        ######

    def mutation(self):  # 变异
        # 单个 多点变异
        variation = np.random.uniform(0, 1)
        if variation < self.config['mutation_rate'] or self.population_like() > 0.8:  # 可设置调节
            mutate_index = np.random.randint(0, int(self.config['population_size'] - 1))
            mutate_position = np.random.randint(0, int(sum(self.weight)) - 1)
            sort = [i for i in range(self.figure)]
            np.random.shuffle(sort)
            self.mutata_num = int(self.figure / 2)
            for i in range(self.mutata_num):
                if self.population[mutate_index][sort[i]] == '0':
                    self.population[mutate_index][sort[i]] = '1'
                else:
                    self.population[mutate_index][sort[i]] = '0'
            ####

    def run(self):  # 自动进行
        fp = open('./log.txt', 'w')
        commutative_algebra = 0
        last_max_fitness = [0 for i in range(19)]

        self.set_fitness(otsu)
        self.init_population()
        fp.write('初始种群：' + str(self.population) + '\n')
        self.fitness_compute()
        last_max_fitness.append(max(self.fitness))
        fp.write('适应度：' + str(self.fitness) + '\n')
        fp.write('最大适应度：' + str(max(self.fitness)) + '所在个体：' + str(self.fitness.index(max(self.fitness))) + '\n')
        while (commutative_algebra < self.config['max_commutative_algebra'] and (
                max(last_max_fitness) - min(last_max_fitness)) > 1e-1):
            last_max_fitness.append(max(self.fitness))
            last_max_fitness.pop(0)
            self.selection()
            self.crossover()
            self.mutation()
            self.fitness_compute()
            commutative_algebra += 1
            fp.write('种群：' + str(self.population) + '\n')
            fp.write('适应度：' + str(self.fitness) + '\n')
            fp.write('最大适应度：' + str(max(self.fitness)) + '所在个体：' + \
                     str(self.fitness.index(max(self.fitness))) + '\n')
        fp.close()

    def population_like(self):  # 种群相同度
        difference = []
        difference_num = []
        for diff in self.population:
            if diff not in difference:
                difference.append(diff)
        for diff in difference:
            difference_num.append(self.population.count(diff))
        difference_r = []
        for i in range(len(difference_num)):
            difference_r.append(difference_num[i] / len(self.population))
        return max(difference_r)



def image_change(image,value):
    image_new = np.array(image)
    for y in range(len(image_new)):
        for x in range(len(image_new[0])):
            if image_new[y][x] > value:
                image_new[y][x] = 255
            else:
                image_new[y][x] = 0
    return Image.fromarray(image_new)

if __name__ == '__main__':
    image = Image.open('./eiffeltower.jpg')
    image_gray = image.convert('L')

    #run
    fp = open('./log.txt', 'w')
    ther = ThresholdSegmentationOfGrayImage(image_gray.histogram(), 8, [1 for i in range(8)])
    commutative_algebra = 0
    last_max_fitness = [0 for i in range(30)]

    ther.set_config({'population_size':10})
    ther.set_fitness(otsu)
    ther.init_population()
    fp.write('初始种群：' + str(ther.population) + '\n')
    ther.fitness_compute()
    last_max_fitness.append(max(ther.fitness))
    fp.write('适应度：' + str(ther.fitness) + '\n')
    fp.write('最大适应度：' + str(max(ther.fitness)) + '所在个体：' + str(ther.fitness.index(max(ther.fitness))) + '\n')
    while (commutative_algebra < ther.config['max_commutative_algebra'] and (
            max(last_max_fitness) - min(last_max_fitness)) > 1e-1):
        last_max_fitness.append(max(ther.fitness))
        last_max_fitness.pop(0)

        # optimal_individual = ther.population[ther.fitness.index(max(ther.fitness))]  # 每轮保留最优个体，效果显著提升
        ther.selection()
        ther.crossover()
        ther.mutation()
        ther.fitness_compute()
        # ther.population[-1] = optimal_individual
        # ther.fitness_compute()
        commutative_algebra += 1
        fp.write('种群：' + str(ther.population) + '\n')
        fp.write('适应度：' + str(ther.fitness) + '\n')
        fp.write('最大适应度：' + str(max(ther.fitness)) + '所在个体：' + \
                 str(ther.fitness.index(max(ther.fitness))) + '\n')
    fp.close()

    fit_max =ther.decode(ther.population[ther.fit.index(max(ther.fit))])
    print(fit_max)
    new_image = image_change(image_gray, fit_max)
    new_image.save(r'./eiffeltower_fit.jpg')