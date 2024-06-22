# TT100K_Cropped

## 背景

这是我《计算机视觉与模式识别》课程 “交通标志的定位与识别” 综合实验训练分类器所用数据集，好像网络上纯交通标志分类的数据集只有德国的[GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html)等，没有中国的交通标志分类的数据集，我就自己从[TT100K](https://cg.cs.tsinghua.edu.cn/traffic-sign/)中提取做了一个。

## 数据集介绍

TT100K_Cropped 是一个中国交通标志分类数据集，专门从 TT100K 数据集中裁剪出目标图像组成。TT100K 数据集由清华大学和腾讯公司合作创建，包含了从 100,000 张腾讯街景全景图中提取的 30,000 个交通标志实例。TT100K_Cropped 通过裁剪这些标志以专注于标志本身，为交通标志检测和分类任务提供了一个更专注的数据集。

```
.
├── DataSet
│   ├── test
│   └── train
├── LICENSE
├── README.md
├── code
│   ├── __pycache__
│   ├── anno_func.py
│   └── example.ipynb   从 TT100K 数据集裁剪目标图像
└── pic
    ├── test_class.png
    ├── train_class.png
    └── u16.jpg
```

## 数据集特点

- **数据来源**: 从 TT100K 数据集中裁剪得到的目标图像。
- **多样性**: 图像涵盖了多种光照和天气条件，具有很高的多样性。
- **类别不平衡**: 数据集各类别样本数不平衡，请格外注意。
- **添加负类**: 添加从 TT100K 数据集的`other`文件夹中随机提取的非交通标志区域

## 数据集类别

![](pic\u16.jpg)

# Table 1
| i1 | i2 | i3 | i4 |
|----|----|----|----|
| Walk 步行 | Non_motorized vehicles 非机动车行驶 | Round the island 环岛行驶 | Motor vehicle 机动车行驶 |

| i5 | i6 | i7 | i8 |
|----|----|----|----|
| Keep on the right side of the road 靠右侧道路行驶 | Keep on the left side of the road 靠左侧道路行驶 | Drive straight and turn right at the grade separation 立体交叉直行和右转弯行驶 | Drive straight and turn left 立体交叉直行和左转弯行驶 |

| i9 | i10 | i11 | i12 |
|----|----|----|----|
| Honk 鸣喇叭 | Turn right 向右转弯 | Turn left and right 向左向右转弯 | Turn left 向左转弯 |

| i13 | i14 | i15 | i150 |
|------|------|------|------|
| One way, straight 直行 | Go straight and turn right 直行和向右转弯 | Go straight and turn left 直行和向左转弯 | Minimum Speed Limit 最低限速（50） |

# Table 2
| ip | p1 | p2 | p3 |
|----|----|----|----|
| Pedestrian Crossing 人行横道 | No Overtaking 超车 | Ban animal-drawn vehicles entering 禁止畜力车进入 | Ban large passenger vehicles from entering 禁止大型客车驶入 |

| p4 | p5 | p6 | p7 |
|----|----|----|----|
| Prohibition of electric tricycles 禁止电动三轮车驶入 | No U-turn 禁止掉头 | Prohibition of non-motorized vehicles 禁止非机动车进入 | No left turn for trucks 禁止载货汽车左转 |

| p8 | p9 | p10 | p11 |
|----|----|----|----|
| It is forbidden to tow or trailer vehicles 禁止汽车拖、挂车驶入 | Prohibit pedestrians entering 禁止行人进入 | Prohibit motorized vehicles 禁止机动车驶入 | Prohibit honking 禁止鸣喇叭 |

| p12 | p13 | p14 | p15 |
|----|----|----|----|
| Two-wheeled motorcycles are prohibited 禁止二轮摩托车驶入 | Prohibit certain two types of vehicles from entering 禁止某两种车驶入 | Prohibition straight 禁止直行 | No rickshaws are allowed 禁止人力车进入 |

# Table 3
| p16 | p17 | p18 | p19p |
|-----|-----|-----|------|
| Prohibition of human-powered cargo tricycles from entering 禁止人力货运三轮车进入 | Prohibition of human-powered cargo tricycles from entering 禁止人力货运三轮车进入 | Prohibition of tricycles and motor vehicles 禁止三轮车机动车通行 | Prohibited right turn 禁止向右转弯 |

| p20 | p21 | p22 | p23 |
|-----|-----|-----|-----|
| No left or right turn 禁止向左向右转弯 | Prohibited right turn and straight 禁止直行和向右转弯 | Prohibition of tricycles and motor vehicles 禁止三轮车机动车通行 | Prohibited left turn 禁止向左转弯 |

| p24 | p25 | p26 | p27 |
|-----|-----|-----|-----|
| Prohibition of right turning of passenger cars 禁止小客车右转 | Prohibition of entry of small passenger cars 禁止小客车驶入 | Prohibit laden car into 禁止载货汽车驶入 | Prohibit entry of vehicles transporting dangerous goods 禁止运输危险物品车辆驶入 |

| p28 | p29 | pa10 | pb |
|------|------|-------|------|
| Prohibited left turn and straight 禁止直行和向左转弯 | Prohibit tractors from entering 禁止拖拉机驶入 | Limit axle load 限制轴重(10t) | No thoroughfare 禁止通行 |

# Table 4
| pc | pd | pe | pg |
|----|----|----|----|
| Parking inspection 停车检查 | Customs 海关 | Give Way to Oncoming Vehicles 会车让行 | Slow down and give way 减速让行 |

| ph3.5 | pl40 | pm10 | pn |
|-------|------|------|----|
| Limit height 限制高度(3.5m) | Limit speed 限制速度(40) | Limit weight 限制质量(10t) | No parking 禁止停车 |

| pne | pnl | pr40 | ps |
|------|------|------|------|
| No Entry 禁止驶入 | No long-term parking 禁止长时停车 | Speed restrictions lifted 解除限制速度 | Park to give way 停车让行 |

| pw3 | w1 | w2 | w3 |
|-----|----|----|----|
| Limit width 限制宽度(3.5m) | Dangerous road near the mountain 傍山险路 | Dangerous road near the mountain 傍山险路 | Village 村庄 |

# Table 5
| w4 | w5 | w6 | w7 |
|----|----|----|----|
| Embankment road 堤坝路 | Embankment road 堤坝路 | T-shaped plane crossing 丁字平面交叉 | Ferry 渡口 |

| w8 | w9 | w10 | w11 |
|----|----|-----|-----|
| Narrow on both sides 两侧变窄 | Watch out for falling rocks 注意落石 | Reverse detour 反向弯路 | Reverse detour 反向弯路 |

| w12 | w13 | w14 | w15 |
|------|------|------|------|
| Manshuiqiao 漫水桥 | Crossroads 十字交叉路口 | Crossroads 十字交叉路口 | Y-shaped intersection Y形交叉路口 |

| w16 | w17 | w18 | w19 |
|------|------|------|------|
| Y-shaped intersection Y形交叉路口 | Y-shaped intersection Y形交叉路口 | Road Narrows on Left 左侧变窄 | Y-shaped intersection Y形交叉路口 |

# Table 6
| w20 | w21 | w22 | w23 |
|------|------|------|------|
| T intersection T形交叉路口 | T intersection T形交叉路口 | T intersection T形交叉路口 | Roundabout 环形交叉路口 |

| w24 | w25 | w26 | w27 |
|------|------|------|------|
| Continuous detour 连续弯路 | Slopes 连续下坡 | Uneven road 路面不平 | Watch out for rain (snow) days 注意雨（雪）天 |

| w28 | w29 | w30 | w31 |
|------|------|------|------|
| Low-lying road 路面低洼 | High road surface 路面高突 | Stroll 慢行 | Up a steep slope 上陡坡 |

| w32 | w33 | w34 | w35 |
|------|------|------|------|
| Construction 施工 | Cross plane 十字平面交叉 | Accident-prone road 事故易发路段 | Two-way traffic 双向交通 |

# Table 7
| w36 | w37 | w38 | w39 |
|------|------|------|------|
| Watch out for wild animals 注意野生动物 | Tunnel 隧道 | Tunnel driving lights 隧道开车灯 | Hump bridge 驼峰桥 |

| w40 | w41 | w42 | w43 |
|------|------|------|------|
| Unguarded railway crossing 无人看守铁路道口 | Down steep slope 下陡坡 | Sharp detour 急弯路 | Sharp detour 急弯路 |

| w44 | w45 | w46 | w47 |
|------|------|------|------|
| Slippery 易滑 | Watch out for semaphores 注意信号灯 | Someone guards the railway crossing 有人看守铁路道口 | Narrowing on the right 右侧变窄 |

| w48 | w49 | w50 | w51 |
|------|------|------|------|
| Detour right 右侧绕行 | Narrow on both sides 两侧变窄 | Pay attention to keeping the distance between cars 注意保持车距 | Pay attention to adverse weather conditions 注意不利气象条件 |

# Table 8
| w52 | w53 | w54 | w55 |
|------|------|------|------|
| Pay attention to the disabled 注意残疾人 | Watch out for tidal lanes 注意潮汐车道 | Pay attention to foggy days 注意雾天 | Pay attention to children 注意儿童 |

| w56 | w57 | w58 | w59 |
|------|------|------|------|
| Pay attention to non-motorized vehicles 注意非机动车 | Pay attention to pedestrians 注意行人 | Pay attention to confluence 注意合流 | Pay attention to confluence 注意合流 |

| w60 | w61 | w62 | w63 |
|-----|-----|-----|-----|
| Crosswinds 注意横风 | Watch out for icing on the road 注意路面结冰 | Watch out for falling rocks 注意落石 | Drive With Caution 注意危险 |

| w64 | w65 | w66 | w67 |
|-----|-----|-----|-----|
| Pay attention to livestock 注意牲畜 | Detour on the left 左侧绕行 | Left and right detour 左右绕行 | Pay attention to the queues of vehicles ahead 注意前方车辆排队 |

# Supplementary Signs
| io | wo | po |
|----|----|----|
| 其他指示标志 Other signs | 其他警告标志 Other warning signs | 其他禁止标志 Other prohibition signs |

## 数据集样本分布

### 训练集

![](pic\train_class.png)

### 测试集

![](pic\test_class.png)

## 许可证

本项目基于 MIT 许可证开源，详细信息请参阅 LICENSE 文件。

## 参考

- [TT100K 数据集](https://cg.cs.tsinghua.edu.cn/traffic-sign/)

- [中国交通标志牌数据集TT100K中的类别ID及其图标罗列以及含义详细介绍](https://blog.csdn.net/qq_37346140/article/details/127581223)