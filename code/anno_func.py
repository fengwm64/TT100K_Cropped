import cv2
import numpy as np
import copy
import json
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

type45="i2,i4,i5,il100,il60,il80,io,ip,p10,p11,p12,p19,p23,p26,p27,p3,p5,p6,pg,ph4,ph4.5,ph5,pl100,pl120,pl20,pl30,pl40,pl5,pl50,pl60,pl70,pl80,pm20,pm30,pm55,pn,pne,po,pr40,w13,w32,w55,w57,w59,wo"
type45 = type45.split(',')

def load_img(annos, datadir, imgid):
    img = annos["imgs"][imgid]
    imgpath = datadir + '/' + img['path']
    imgdata = cv2.imread(imgpath)
    if imgdata.max() > 2:
        imgdata = imgdata / 255.
    return imgdata

def load_mask(annos, datadir, imgid, imgdata):
    img = annos["imgs"][imgid]
    mask = np.zeros(imgdata.shape[:-1])
    mask_poly = np.zeros(imgdata.shape[:-1])
    mask_ellipse = np.zeros(imgdata.shape[:-1])
    for obj in img['objects']:
        box = obj['bbox']
        cv2.rectangle(mask, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), 1, -1)
        if 'polygon' in obj and len(obj['polygon']) > 0:
            pts = np.array(obj['polygon'])
            cv2.fillPoly(mask_poly, [pts.astype(np.int32)], 1)
        else:
            cv2.rectangle(mask_poly, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), 1, -1)
        if 'ellipse' in obj:
            rbox = obj['ellipse']
            rbox = ((rbox[0][0], rbox[0][1]), (rbox[1][0], rbox[1][1]), rbox[2])
            cv2.ellipse(mask_ellipse, rbox, 1, -1)
        else:
            cv2.rectangle(mask_ellipse, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), 1, -1)
    mask = np.multiply(np.multiply(mask, mask_poly), mask_ellipse)
    return mask
    
def draw_all(annos, datadir, imgid, imgdata, color=(0,1,0), have_mask=True, have_label=True):
    img = annos["imgs"][imgid]
    if have_mask:
        mask = load_mask(annos, datadir, imgid, imgdata)
        imgdata = imgdata.copy()
        imgdata[:,:,0] = np.clip(imgdata[:,:,0] + mask*0.7, 0, 1)
    for obj in img['objects']:
        box = obj['bbox']
        cv2.rectangle(imgdata, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), color, 3)
        ss = obj['category']
        if 'correct_catelog' in obj:
            ss = ss + '->' + obj['correct_catelog']
        if have_label:
            cv2.putText(imgdata, ss, (int(box['xmin']), int(box['ymin']) - 10), 0, 1, color, 2)
    return imgdata

def rect_cross(rect1, rect2):
    rect = [max(rect1[0], rect2[0]),
            max(rect1[1], rect2[1]),
            min(rect1[2], rect2[2]),
            min(rect1[3], rect2[3])]
    rect[2] = max(rect[2], rect[0])
    rect[3] = max(rect[3], rect[1])
    return rect

def rect_area(rect):
    return float(max(0.0, (rect[2] - rect[0]) * (rect[3] - rect[1])))

def calc_cover(rect1, rect2):
    crect = rect_cross(rect1, rect2)
    return rect_area(crect) / rect_area(rect2)

def calc_iou(rect1, rect2):
    crect = rect_cross(rect1, rect2)
    ac = rect_area(crect)
    a1 = rect_area(rect1)
    a2 = rect_area(rect2)
    return ac / (a1 + a2 - ac)

def get_refine_rects(annos, raw_rects, minscore=20):
    cover_th = 0.5
    refine_rects = {}

    for imgid in raw_rects.keys():
        v = raw_rects[imgid]
        tv = copy.deepcopy(sorted(v, key=lambda x:-x[2]))
        nv = []
        for obj in tv:
            rect = obj[1]
            rect[2] += rect[0]
            rect[3] += rect[1]
            if rect_area(rect) == 0:
                continue
            if obj[2] < minscore:
                continue
            cover_area = 0
            for obj2 in nv:
                cover_area += calc_cover(obj2[1], rect)
            if cover_area < cover_th:
                nv.append(obj)
        refine_rects[imgid] = nv
    results = {}
    for imgid, v in refine_rects.items():
        objs = []
        for obj in v:
            mobj = {"bbox": {"xmin": obj[1][0], "ymin": obj[1][1], "xmax": obj[1][2], "ymax": obj[1][3]},
                    "category": annos['types'][int(obj[0] - 1)], "score": obj[2]}
            objs.append(mobj)
        results[imgid] = {"objects": objs}
    results_annos = {"imgs": results}
    return results_annos

def box_long_size(box):
    return max(box['xmax'] - box['xmin'], box['ymax'] - box['ymin'])

def eval_annos(annos_gd, annos_rt, iou=0.75, imgids=None, check_type=True, types=None, minscore=40, minboxsize=0, maxboxsize=400, match_same=True):
    ac_n, ac_c = 0, 0
    rc_n, rc_c = 0, 0
    if imgids == None:
        imgids = annos_rt['imgs'].keys()
    if types != None:
        types = {t: 0 for t in types}
    miss = {"imgs": {}}
    wrong = {"imgs": {}}
    right = {"imgs": {}}
    
    for imgid in imgids:
        v = annos_rt['imgs'][imgid]
        vg = annos_gd['imgs'][imgid]
        convert = lambda objs: [[obj['bbox'][key] for key in ['xmin', 'ymin', 'xmax', 'ymax']] for obj in objs]
        objs_g = vg["objects"]
        objs_r = v["objects"]
        bg = convert(objs_g)
        br = convert(objs_r)
        
        match_g = [-1] * len(bg)
        match_r = [-1] * len(br)
        if types != None:
            for i in range(len(match_g)):
                if objs_g[i]['category'] not in types:
                    match_g[i] = -2
            for i in range(len(match_r)):
                if 'score' in objs_r[i] and objs_r[i]['score'] < minscore:
                    match_r[i] = -2
        matches = []
        for i, boxg in enumerate(bg):
            for j, boxr in enumerate(br):
                if match_g[i] == -2 or match_r[j] == -2:
                    continue
                if match_same and objs_g[i]['category'] != objs_r[j]['category']:
                    continue
                tiou = calc_iou(boxg, boxr)
                if tiou > iou:
                    matches.append((tiou, i, j))
        matches = sorted(matches, key=lambda x: -x[0])
        for tiou, i, j in matches:
            if match_g[i] == -1 and match_r[j] == -1:
                match_g[i] = j
                match_r[j] = i
                
        for i in range(len(match_g)):
            boxsize = box_long_size(objs_g[i]['bbox'])
            erase = False
            if not (minboxsize <= boxsize < maxboxsize):
                erase = True
            if erase:
                if match_g[i] >= 0:
                    match_r[match_g[i]] = -2
                match_g[i] = -2
        
        for i in range(len(match_r)):
            boxsize = box_long_size(objs_r[i]['bbox'])
            if match_r[i] != -1:
                continue
            if not (minboxsize <= boxsize < maxboxsize):
                match_r[i] = -2
                    
        miss["imgs"][imgid] = {"objects": []}
        wrong["imgs"][imgid] = {"objects": []}
        right["imgs"][imgid] = {"objects": []}
        miss_objs = miss["imgs"][imgid]["objects"]
        wrong_objs = wrong["imgs"][imgid]["objects"]
        right_objs = right["imgs"][imgid]["objects"]
        
        tt = 0
        for i in range(len(match_g)):
            if match_g[i] == -1:
                miss_objs.append(objs_g[i])
        for i in range(len(match_r)):
            if match_r[i] == -1:
                obj = copy.deepcopy(objs_r[i])
                obj['correct_catelog'] = 'none'
                wrong_objs.append(obj)
            elif match_r[i] != -2:
                j = match_r[i]
                obj = copy.deepcopy(objs_r[i])
                if not check_type or objs_g[j]['category'] == objs_r[i]['category']:
                    right_objs.append(objs_r[i])
                    tt += 1
                else:
                    obj['correct_catelog'] = objs_g[j]['category']
                    wrong_objs.append(obj)
                    
        
        rc_n += len(objs_g) - match_g.count(-2)
        ac_n += len(objs_r) - match_r.count(-2)
        
        ac_c += tt
        rc_c += tt
    if types == None:
        styps = "all"
    elif len(types) == 1:
        styps = next(iter(types.keys()))
    elif not check_type or len(types) == 0:
        styps = "none"
    else:
        styps = "[%s, ...total %s...]" % (next(iter(types.keys())), len(types))
    report = "iou:%s, size:[%s,%s), types:%s, accuracy:%s, recall:%s" % (
        iou, minboxsize, maxboxsize, styps, 1 if ac_n == 0 else ac_c * 1.0 / ac_n, 1 if rc_n == 0 else rc_c * 1.0 / rc_n)
    summary = {
        "iou": iou,
        "accuracy": 1 if ac_n == 0 else ac_c * 1.0 / ac_n,
        "recall": 1 if rc_n == 0 else rc_c * 1.0 / rc_n,
        "miss": miss,
        "wrong": wrong,
        "right": right,
        "report": report
    }
    return summary

def crop_and_save_images(annos, datadir, img_ids, output_dir, num_per_class):
    """
    裁剪并保存图像函数。

    参数：
    - annos：包含注释数据的字典。
    - datadir：数据目录的路径，包含注释文件和图像文件夹。
    - img_ids：要处理的图像标识列表。
    - output_dir：保存裁剪图像的目标目录。
    - num_per_class：每个类别保存的最大图像数，默认为200。

    返回值：
    无，直接保存裁剪后的图像到指定目录。

    注意：
    - 函数假定使用了cv2和numpy库。
    - 裁剪后的图像将按类别存储在output_dir下的子目录中。
    - 如果某些图像或对象的边界框无效（例如超出图像边界），将跳过处理。

    """

    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化每个类别的计数器
    class_count = {cls: 0 for cls in annos['types']}

    # 遍历每个图像标识
    for imgid in img_ids:
        # 加载图像数据
        imgdata = load_img(annos, datadir, imgid)
        
        # 检查图像是否成功加载
        if imgdata is None or imgdata.size == 0:
            continue
        
        # 获取图像的对象注释
        img_annotations = annos['imgs'][imgid]['objects']
        
        # 遍历图像中的每个对象注释
        for obj in img_annotations:
            cls = obj['category']  # 获取对象类别
            # 检查当前类别保存的图像数量是否达到上限
            if class_count[cls] < num_per_class:
                bbox = obj['bbox']  # 获取对象的边界框坐标
                x1, y1, x2, y2 = map(int, [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']])
                
                # 确保边界框坐标在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(imgdata.shape[1], x2), min(imgdata.shape[0], y2)
                
                # 如果边界框不合法，则跳过当前对象
                if x1 >= x2 or y1 >= y2:
                    print(f"图像 {imgid} 的边界框坐标无效，跳过处理。")
                    continue
                
                # 根据边界框裁剪图像
                cropped_img = imgdata[y1:y2, x1:x2]
                
                # 如果裁剪后的图像数据类型为浮点数，则转换为 uint8
                if cropped_img.dtype == np.float64 or cropped_img.dtype == np.float32:
                    cropped_img = (cropped_img * 255).astype(np.uint8)
                elif cropped_img.dtype == np.uint64 or cropped_img.dtype == np.uint32:
                    cropped_img = cropped_img.astype(np.uint8)
                
                # 按类别创建存储图像的子目录
                class_dir = os.path.join(output_dir, cls)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                
                # 图像文件名格式：图像标识_类别计数.png
                img_filename = f"{imgid}_{class_count[cls]}.png"
                img_path = os.path.join(class_dir, img_filename)
                
                # 使用 OpenCV 保存图像
                try:
                    cv2.imwrite(img_path, cropped_img)
                    print(f"保存图像 {img_path}")
                except Exception as e:
                    print(f"保存图像 {img_path} 时出错：{str(e)}")
                
                # 更新当前类别保存的图像数量
                class_count[cls] += 1
                
                # 如果当前类别保存的图像数量达到上限，则跳出循环
                if class_count[cls] >= num_per_class:
                    break
                
def crop_and_save_images(annos, datadir, img_ids, output_dir):
    """
    裁剪并保存图像函数。

    参数：
    - annos：包含注释数据的字典。
    - datadir：数据目录的路径，包含注释文件和图像文件夹。
    - img_ids：要处理的图像标识列表。
    - output_dir：保存裁剪图像的目标目录。

    返回值：
    无，直接保存裁剪后的图像到指定目录。

    注意：
    - 函数假定使用了cv2和numpy库。
    - 裁剪后的图像将按类别存储在output_dir下的子目录中。
    - 如果某些图像或对象的边界框无效（例如超出图像边界），将跳过处理。
    """

    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化每个类别的计数器
    class_count = {cls: 0 for cls in annos['types']}

    # 使用 tqdm 显示进度条
    pbar = tqdm(total=len(img_ids), desc="Processing images")

    # 遍历每个图像标识
    for imgid in img_ids:
        # 加载图像数据
        imgdata = load_img(annos, datadir, imgid)
        
        # 检查图像是否成功加载
        if imgdata is None or imgdata.size == 0:
            pbar.update(1)
            continue
        
        # 获取图像的对象注释
        img_annotations = annos['imgs'][imgid]['objects']
        
        # 遍历图像中的每个对象注释
        for obj in img_annotations:
            cls = obj['category']  # 获取对象类别
            
            bbox = obj['bbox']  # 获取对象的边界框坐标
            x1, y1, x2, y2 = map(int, [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']])
            
            # 确保边界框坐标在图像范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(imgdata.shape[1], x2), min(imgdata.shape[0], y2)
            
            # 如果边界框不合法，则跳过当前对象
            if x1 >= x2 or y1 >= y2:
                pbar.update(1)
                continue
            
            # 根据边界框裁剪图像
            cropped_img = imgdata[y1:y2, x1:x2]
            
            # 如果裁剪后的图像数据类型为浮点数，则转换为 uint8
            if cropped_img.dtype == np.float64 or cropped_img.dtype == np.float32:
                cropped_img = (cropped_img * 255).astype(np.uint8)
            elif cropped_img.dtype == np.uint64 or cropped_img.dtype == np.uint32:
                cropped_img = cropped_img.astype(np.uint8)
            
            # 按类别创建存储图像的子目录
            class_dir = os.path.join(output_dir, cls)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            
            # 图像文件名格式：图像标识_类别计数.png
            img_filename = f"{imgid}_{class_count[cls]}.png"
            img_path = os.path.join(class_dir, img_filename)
            
            # 使用 OpenCV 保存图像
            try:
                cv2.imwrite(img_path, cropped_img)
                pbar.update(1)
            except Exception as e:
                print(f"保存图像 {img_path} 时出错：{str(e)}")
            
            # 更新当前类别保存的图像数量
            class_count[cls] += 1
        
        pbar.set_postfix({cls: count for cls, count in class_count.items()})  # 更新进度条的后缀信息
    
    pbar.close()  # 关闭进度条

    # 显示每个类别保存的图像数量
    for cls, count in class_count.items():
        print(f"类别 {cls} 保存了 {count} 张图像。")
        
    return class_count

def random_crop_negative_class(input_dir, output_dir):
    """
    从指定文件夹中随机读取图片，并随机裁剪一部分作为负类，并保存到指定目录。

    参数：
    - input_dir：输入文件夹，包含要读取的图片文件。
    - output_dir：输出文件夹，保存裁剪后的负类图片。

    返回值：
    无，直接保存裁剪后的负类图片到指定目录。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取输入目录下所有图片文件的路径列表
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # 如果没有找到图片文件，则直接返回
    if not image_files:
        print(f"未在目录 {input_dir} 下找到任何图片文件。")
        return

    # 随机选择一张图片
    img_path = random.choice(image_files)

    # 读取图片
    img = cv2.imread(img_path)

    # 获取图片的高度和宽度
    height, width = img.shape[:2]

    # 随机生成裁剪区域的起始点
    start_x = random.randint(0, max(0, width - 256))
    start_y = random.randint(0, max(0, height - 256))

    # 随机生成裁剪区域的结束点
    end_x = min(start_x + 256, width)
    end_y = min(start_y + 256, height)

    # 裁剪图片
    cropped_img = img[start_y:end_y, start_x:end_x]

    # 生成保存路径
    filename = os.path.basename(img_path)
    save_path = os.path.join(output_dir, filename)

    # 保存裁剪后的图片
    cv2.imwrite(save_path, cropped_img)

    print(f"已保存裁剪后的负类图片到 {save_path}")
    
def plot_class_distribution(class_count):
    # 提取类别和数量
    classes = list(class_count.keys())
    counts = list(class_count.values())

    # 绘制直方图
    plt.figure(figsize=(40, 10))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution of Cropped Images')
    plt.xticks(rotation=45, fontsize='small')  # 调整刻度的旋转角度和字体大小
    plt.tight_layout()
    plt.savefig("1.png",dpi=800)