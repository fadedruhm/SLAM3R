import os
import re

def rename_images(directory):
    """
    批量重命名目录中的图片文件，使其符合 SLAM3R 的要求。
    假设文件名格式如 DJI_20260315063032_0080_D.jpg，
    提取数字部分（如 0080）并重命名为 frame_0080.jpg。
    """
    # 支持的图片扩展名
    supported_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.heif']

    # 遍历目录中的文件
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # 检查是否是文件且是支持的图片格式
        if os.path.isfile(filepath) and any(filename.lower().endswith(ext) for ext in supported_extensions):
            # 使用正则表达式提取数字部分（假设数字在文件名末尾，如 _0080_D）
            # 匹配最后一个下划线后的数字
            match = re.search(r'_(\d+)_D', filename)
            if match:
                number = match.group(1)  # 提取数字，如 '0080'
                # 获取文件扩展名
                _, ext = os.path.splitext(filename)
                # 新文件名
                new_filename = f"frame_{number}{ext}"
                new_filepath = os.path.join(directory, new_filename)

                # 检查新文件名是否已存在，避免覆盖
                if os.path.exists(new_filepath):
                    print(f"警告: {new_filename} 已存在，跳过 {filename}")
                    continue

                # 重命名文件
                os.rename(filepath, new_filepath)
                print(f"重命名: {filename} -> {new_filename}")
            else:
                print(f"跳过: {filename} (未找到数字部分)")

# 使用示例
if __name__ == "__main__":
    # 将 'your_image_directory' 替换为你的图片目录路径
    image_directory = r"F:\学校、个人相关\大学课程\毕业设计\毕设-cg\素材\images\可见光2"  # 示例路径，请修改为实际路径
    rename_images(image_directory)