# 导入 argparse，用于从命令行读取参数
import argparse

# 从 pathlib 导入 Path，用于更方便地处理文件路径
from pathlib import Path

# 导入 OpenCV，用于读取、缩放、保存图像
import cv2

# 导入 numpy，用于数组处理和保存 .npy 文件
import numpy as np

# 导入 tqdm，用于显示处理进度条
from tqdm import tqdm


# 定义允许被识别为图像文件的后缀集合
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}


# 判断某个路径是否是图像文件
def is_image_file(path: Path) -> bool:
    # 将文件后缀转成小写，然后判断是否在允许的图像后缀集合中
    return path.suffix.lower() in IMG_EXTENSIONS


# 读取灰度图，并 resize 到指定大小
def read_gray(path: Path, size: int):
    # 使用 OpenCV 以灰度模式读取图像
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    # 如果图像读取失败，返回 None
    if image is None:
        return None

    # 将图像 resize 成 size × size，使用三次插值，适合连续灰度图像
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)


# 读取标签图像，并 resize 到指定大小
def read_label(path: Path, size: int):
    # 使用 OpenCV 以灰度模式读取标签图像
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    # 如果标签读取失败，返回 None
    if image is None:
        return None

    # 将标签 resize 成 size × size，使用最近邻插值，避免标签边界被插值污染
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)

    # 对标签进行二值化，大于 127 的像素设为 255，否则设为 0
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 返回二值化后的标签图
    return image


# 尝试读取图像；如果文件不存在或读取失败，则返回全 0 图像
def read_or_zero(path: Path, size: int):
    # 如果路径不存在，返回一个 size × size 的全 0 图像
    if not path.exists():
        return np.zeros((size, size), dtype=np.uint8)

    # 如果路径存在，则按照灰度图读取
    image = read_gray(path, size)

    # 如果读取失败，也返回全 0 图像
    if image is None:
        return np.zeros((size, size), dtype=np.uint8)

    # 如果读取成功，返回读取到的图像
    return image


# 根据切片名，在标签目录中寻找对应的标签文件
def find_label(label_root: Path, slice_stem: str):
    # 如果标签根目录不存在，直接返回 None
    if not label_root.exists():
        return None

    # 构造一组可能的标签文件路径
    candidates = [
        label_root / f"{slice_stem}.png",   # 与切片同名的 png 标签
        label_root / f"{slice_stem}.jpg",   # 与切片同名的 jpg 标签
        label_root / f"{slice_stem}.jpeg",  # 与切片同名的 jpeg 标签
        label_root / f"{slice_stem}.bmp",   # 与切片同名的 bmp 标签
    ]

    # 遍历候选标签路径
    for candidate in candidates:
        # 如果候选文件存在，则返回该标签路径
        if candidate.exists():
            return candidate

    # 如果直接匹配失败，则遍历标签目录下的所有文件
    for path in label_root.iterdir():
        # 判断当前文件是否是图像文件，并且文件名是否等于 slice_stem 或 slice_stem_mask
        if is_image_file(path) and path.stem in {slice_stem, f"{slice_stem}_mask"}:
            # 如果匹配成功，返回该标签路径
            return path

    # 如果没有找到对应标签，返回 None
    return None


# 读取某一张切片对应的所有 DCE 时相图像，并堆叠成一个多通道张量
def read_phase_stack(img_patient_dir: Path, slice_name: str, size: int):
    # 创建一个列表，用来保存不同 phase 的图像
    channels = []

    # 读取平扫/原始 VIBRANT 图像，作为第 0 个通道
    pre = read_gray(img_patient_dir / "VIBRANT" / slice_name, size)

    # 如果 VIBRANT 图像不存在或读取失败，则该切片无效，返回 None
    if pre is None:
        return None

    # 将 VIBRANT 图像加入通道列表
    channels.append(pre)

    # 依次读取 VIBRANT+C1 到 VIBRANT+C8 共 8 个增强时相
    channels.extend(
        read_or_zero(img_patient_dir / f"VIBRANT+C{i}" / slice_name, size)
        for i in range(1, 9)
    )

    # 依次读取 SUB1 到 SUB8 共 8 个减影图像
    channels.extend(
        read_or_zero(img_patient_dir / f"SUB{i}" / slice_name, size)
        for i in range(1, 9)
    )

    # 将所有通道堆叠成 numpy 数组，形状为 [T, H, W]
    # T = 17，其中：
    # 0 是 VIBRANT
    # 1-8 是 VIBRANT+C1 到 VIBRANT+C8
    # 9-16 是 SUB1 到 SUB8
    return np.stack(channels, axis=0)


# 根据当前切片位置 z，获取它附近的 num_slices 个邻近切片索引
def neighbor_indices(z: int, count: int, num_slices: int):
    # 计算当前切片左右各取多少张
    half = num_slices // 2

    # 创建列表，用于保存邻近切片索引
    indices = []

    # 从 -half 到 +half 遍历邻域偏移
    for offset in range(-half, half + 1):
        # 计算邻近切片索引，同时用 min/max 防止越界
        indices.append(min(max(z + offset, 0), count - 1))

    # 返回邻近切片索引列表
    return indices


# 处理单个患者的数据
def process_patient(
    img_patient_dir: Path,   # 当前患者的图像目录
    lbl_patient_dir: Path,   # 当前患者的标签目录
    out_data_dir: Path,      # 输出图像数据的目录
    out_gt_dir: Path,        # 输出标签 GT 的目录
    size: int,               # resize 后的图像大小
    label_phase: str,        # 使用哪个 phase 文件夹下的标签
    num_slices: int,         # 2.5D 输入中包含多少张邻近切片
):
    # 当前患者 VIBRANT 图像目录
    pre_dir = img_patient_dir / "VIBRANT"

    # 当前患者对应的标签目录，例如 labels/患者名/VIBRANT
    label_dir = lbl_patient_dir / label_phase

    # 如果 VIBRANT 图像目录或标签目录不存在，则跳过该患者
    if not pre_dir.exists() or not label_dir.exists():
        return

    # 获取 VIBRANT 目录下所有图像切片，并排序
    slices = sorted([p for p in pre_dir.iterdir() if is_image_file(p)])

    # 如果没有任何切片，则跳过该患者
    if not slices:
        return

    # =========================
    # 第一步：缓存当前患者的所有切片数据
    # =========================

    # phase_cache 用来保存当前患者每一张切片的多时相图像
    # phase_cache 的整体结构可以理解为：
    # [
    #   slice0_phase_stack,   # [T,H,W]
    #   slice1_phase_stack,   # [T,H,W]
    #   slice2_phase_stack,   # [T,H,W]
    #   ...
    # ]
    #
    # 其中每个 phase_stack 的形状是：
    # [T,H,W]
    #
    # T = 17 个通道/时相：
    # [
    #   0: VIBRANT,
    #   1: VIBRANT+C1,
    #   2: VIBRANT+C2,
    #   ...
    #   8: VIBRANT+C8,
    #   9: SUB1,
    #   10: SUB2,
    #   ...
    #   16: SUB8
    # ]
    #
    # H,W = resize 后的图像大小，例如 [256,256]
    phase_cache = []


    # label_cache 用来保存当前患者每一张切片的标签
    # label_cache 的整体结构可以理解为：
    # [
    #   slice0_label,   # [H,W]
    #   slice1_label,   # [H,W]
    #   slice2_label,   # [H,W]
    #   ...
    # ]
    #
    # 每个 label 是一张二值 mask：
    # [H,W]
    #
    # 像素值通常是：
    # 0   = 背景
    # 255 = 肿瘤区域
    label_cache = []


    # 遍历当前患者的所有切片
    # slices 的结构：
    # [
    #   Path("xxx/001.png"),
    #   Path("xxx/002.png"),
    #   Path("xxx/003.png"),
    #   ...
    # ]
    for slice_path in slices:

        # 读取当前切片的所有 DCE phase
        #
        # 输入：
        # 当前患者目录 img_patient_dir
        # 当前切片文件名 slice_path.name，例如 "001.png"
        #
        # 输出：
        # phase_stack: [T,H,W]
        #
        # 举例：
        # phase_stack.shape = [17,256,256]
        #
        # 其中：
        # phase_stack[0]  = VIBRANT 图像
        # phase_stack[1]  = VIBRANT+C1 图像
        # phase_stack[8]  = VIBRANT+C8 图像
        # phase_stack[9]  = SUB1 图像
        # phase_stack[16] = SUB8 图像
        phase_stack = read_phase_stack(img_patient_dir, slice_path.name, size)

        # 根据当前切片名寻找对应标签文件
        #
        # slice_path.stem 表示不带后缀的文件名
        # 例如：
        # slice_path.name = "001.png"
        # slice_path.stem = "001"
        #
        # label_path 可能是：
        # labels/patient001/VIBRANT/001.png
        # labels/patient001/VIBRANT/001_mask.png
        # 或者 None
        label_path = find_label(label_dir, slice_path.stem)

        # 如果找到了标签路径，则读取标签
        #
        # label 的形状：
        # [H,W]
        #
        # 举例：
        # label.shape = [256,256]
        #
        # 如果没找到标签，则 label = None
        label = read_label(label_path, size) if label_path is not None else None

        # 将当前切片的多时相图像加入缓存
        #
        # phase_cache 追加之后类似：
        # [
        #   [T,H,W],
        #   [T,H,W],
        #   [T,H,W],
        #   ...
        # ]
        phase_cache.append(phase_stack)

        # 将当前切片的标签加入缓存
        #
        # label_cache 追加之后类似：
        # [
        #   [H,W],
        #   [H,W],
        #   [H,W],
        #   ...
        # ]
        label_cache.append(label)


    # =========================
    # 第二步：构造 2.5D 样本
    # =========================

    # 遍历当前患者的所有切片
    #
    # enumerate(slices) 会同时得到：
    # z          当前切片的索引，例如 0,1,2,3...
    # slice_path 当前切片路径
    #
    # 假设当前患者有 5 张切片：
    # slices = [
    #   0: 001.png,
    #   1: 002.png,
    #   2: 003.png,
    #   3: 004.png,
    #   4: 005.png
    # ]
    for z, slice_path in enumerate(slices):

        # 如果当前中心切片的图像无效，或者当前中心切片没有标签，则跳过
        #
        # 注意：
        # 只有中心切片必须有 label
        # 邻居切片不需要 label，因为最终监督的是中心切片的 mask
        if phase_cache[z] is None or label_cache[z] is None:
            continue

        # stacks 用来保存当前中心切片附近的 K 张切片
        #
        # 如果 num_slices = 3，那么 K=3
        # stacks 最终结构是：
        # [
        #   z-1 切片的 phase_stack,   # [T,H,W]
        #   z   切片的 phase_stack,   # [T,H,W]
        #   z+1 切片的 phase_stack,   # [T,H,W]
        # ]
        #
        # 所以 stacks 可以理解为：
        # [
        #   [T,H,W],
        #   [T,H,W],
        #   [T,H,W]
        # ]
        stacks = []

        # 获取当前切片 z 的邻近切片索引
        #
        # 如果 num_slices = 3，表示取 [z-1, z, z+1]
        #
        # 例如当前 z = 2：
        # neighbor_indices(2, 5, 3) -> [1,2,3]
        #
        # 表示使用：
        # slice1, slice2, slice3
        #
        # 如果当前 z = 0：
        # neighbor_indices(0, 5, 3) -> [0,0,1]
        #
        # 因为 z-1 越界，所以用第 0 张切片补齐
        #
        # 如果当前 z = 4：
        # neighbor_indices(4, 5, 3) -> [3,4,4]
        #
        # 因为 z+1 越界，所以用最后一张切片补齐
        for neighbor in neighbor_indices(z, len(slices), num_slices):

            # 取出邻近切片的多时相图像
            #
            # stack 的形状：
            # [T,H,W]
            #
            # 举例：
            # stack.shape = [17,256,256]
            stack = phase_cache[neighbor]

            # 如果邻近切片读取失败，则用当前中心切片代替
            #
            # 这样可以保证 stacks 里面每个元素都是有效的 [T,H,W]
            if stack is None:
                stack = phase_cache[z]

            # 将邻近切片加入 stacks
            #
            # append 一次后：
            # stacks = [
            #   [T,H,W]
            # ]
            #
            # append 三次后：
            # stacks = [
            #   [T,H,W],
            #   [T,H,W],
            #   [T,H,W]
            # ]
            stacks.append(stack)

        # 将 K 张邻近切片堆叠起来
        #
        # 原来：
        # stacks = [
        #   [T,H,W],
        #   [T,H,W],
        #   [T,H,W]
        # ]
        #
        # np.stack(stacks, axis=0) 之后：
        # x = [K,T,H,W]
        #
        # 如果 num_slices = 3：
        # x.shape = [3,17,256,256]
        #
        # 含义是：
        # x[0] = 前一张切片的 17 个时相图像，形状 [17,256,256]
        # x[1] = 当前中心切片的 17 个时相图像，形状 [17,256,256]
        # x[2] = 后一张切片的 17 个时相图像，形状 [17,256,256]
        #
        # 更细一点：
        # x[1,0]  = 当前中心切片的 VIBRANT 图像
        # x[1,1]  = 当前中心切片的 VIBRANT+C1 图像
        # x[1,8]  = 当前中心切片的 VIBRANT+C8 图像
        # x[1,9]  = 当前中心切片的 SUB1 图像
        # x[1,16] = 当前中心切片的 SUB8 图像
        x = np.stack(stacks, axis=0)

        # 构造输出文件名
        #
        # 例如：
        # img_patient_dir.name = "patient001"
        # slice_path.stem = "003"
        #
        # out_name = "patient001_003"
        out_name = f"{img_patient_dir.name}_{slice_path.stem}"

        # 保存 2.5D 图像数据
        #
        # 保存内容：
        # x: [K,T,H,W]
        #
        # 举例：
        # processed_25d_dce/train/data/patient001_003.npy
        #
        # 里面的数据形状：
        # [3,17,256,256]
        np.save(str(out_data_dir / f"{out_name}.npy"), x)

        # 保存中心切片的标签
        #
        # 注意：
        # 输入 x 是 [K,T,H,W]
        # 但是标签只保存中心切片 z 的 mask
        #
        # label_cache[z] 的形状：
        # [H,W]
        #
        # 举例：
        # processed_25d_dce/train/GT/patient001_003.png
        #
        # 里面是：
        # 当前中心切片 patient001_003 的肿瘤分割标签
        cv2.imwrite(str(out_gt_dir / f"{out_name}.png"), label_cache[z])


# 主函数，负责解析命令行参数并批量处理 train/val/test
def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Build 2.5D BreastDM DCE samples.")

    # 添加数据集根目录参数，必须提供
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root containing train/val/test folders."
    )

    # 添加输出目录参数，默认输出到 ./processed_25d_dce
    parser.add_argument(
        "--output_root",
        type=str,
        default="./processed_25d_dce"
    )

    # 添加图像 resize 大小参数，默认 256
    parser.add_argument(
        "--size",
        type=int,
        default=256
    )

    # 添加标签 phase 参数，默认使用 VIBRANT 文件夹下的标签
    parser.add_argument(
        "--label_phase",
        type=str,
        default="VIBRANT",
        help="Label folder to use, e.g. VIBRANT or SUB2."
    )

    # 添加 2.5D 邻近切片数量参数，默认 3，即 [z-1,z,z+1]
    parser.add_argument(
        "--num_slices",
        type=int,
        default=3,
        help="Odd number of neighboring slices, default [z-1,z,z+1]."
    )

    # 解析命令行传入的参数
    args = parser.parse_args()

    # 判断 num_slices 是否为正奇数
    if args.num_slices < 1 or args.num_slices % 2 == 0:
        # 如果不是正奇数，则抛出错误
        raise ValueError("--num_slices must be a positive odd number.")

    # 将数据集根目录转换为 Path 对象
    root = Path(args.dataset_root)

    # 将输出根目录转换为 Path 对象
    output_root = Path(args.output_root)

    # 依次处理 train、val、test 三个数据划分
    for split in ["train", "val", "test"]:
        # 当前 split 的图像目录，例如 root/train/images
        split_img_dir = root / split / "images"

        # 当前 split 的标签目录，例如 root/train/labels
        split_lbl_dir = root / split / "labels"

        # 如果图像目录或标签目录不存在，则跳过该 split
        if not split_img_dir.exists() or not split_lbl_dir.exists():
            # 打印提示信息
            print(f"[Info] skip {split}: missing images or labels directory")
            continue

        # 当前 split 输出图像数据的目录
        out_data_dir = output_root / split / "data"

        # 当前 split 输出标签 GT 的目录
        out_gt_dir = output_root / split / "GT"

        # 创建输出图像数据目录，如果父目录不存在也一起创建
        out_data_dir.mkdir(parents=True, exist_ok=True)

        # 创建输出标签目录，如果父目录不存在也一起创建
        out_gt_dir.mkdir(parents=True, exist_ok=True)

        # 获取当前 split 下所有患者文件夹
        patients = [p for p in split_img_dir.iterdir() if p.is_dir()]

        # 使用 tqdm 显示患者处理进度
        for patient_dir in tqdm(patients, desc=f"Processing {split}"):
            # 处理当前患者
            process_patient(
                patient_dir,                         # 当前患者图像目录
                split_lbl_dir / patient_dir.name,     # 当前患者标签目录
                out_data_dir,                         # 输出 data 目录
                out_gt_dir,                           # 输出 GT 目录
                args.size,                            # 图像 resize 大小
                args.label_phase,                     # 标签 phase 文件夹
                args.num_slices,                      # 2.5D 邻近切片数量
            )

    # 所有 split 处理完成后，打印保存路径
    print(f"Saved 2.5D samples to {output_root}")

    # 打印样本布局说明
    print("Sample layout: [K,T,H,W], T: 0=VIBRANT, 1-8=VIBRANT+C1..C8, 9-16=SUB1..SUB8")


# Python 程序入口
# 当该文件被直接运行时，执行 main()
if __name__ == "__main__":
    main()