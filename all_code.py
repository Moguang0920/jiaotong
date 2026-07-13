import os

# 配置区，按需修改
ROOT_DIR = r"C:\Users\25838\Desktop\Jiaotong-gpt"
OUTPUT_FILE = "all_code_content.txt"
# 需要提取内容的代码后缀
TARGET_EXTS = {".py", ".js", ".html", ".css", ".json"}
# 忽略的文件夹，不会进入遍历
IGNORE_FOLDERS = {".venv", "node_modules", "__pycache__"}

# 存储所有匹配文件
match_files = []

# 第一步：遍历收集所有目标文件
for dir_path, dir_names, file_names in os.walk(ROOT_DIR):
    # 过滤忽略文件夹，原地修改dir_names实现跳过
    dir_names[:] = [d for d in dir_names if d not in IGNORE_FOLDERS]
    for file_name in file_names:
        ext = os.path.splitext(file_name)[1]
        if ext in TARGET_EXTS:
            full_file_path = os.path.join(dir_path, file_name)
            match_files.append(full_file_path)

# 第二步：写入文件，路径+分隔线+文件内容
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    out_f.write(f"项目根目录：{ROOT_DIR}\n")
    out_f.write(f"总计代码文件数量：{len(match_files)}\n")
    out_f.write("=" * 80 + "\n\n")

    for idx, file_path in enumerate(match_files, 1):
        try:
            # 读取文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # 部分文件编码不是utf8，尝试gbk
            with open(file_path, "r", encoding="gbk") as f:
                content = f.read()
        except Exception as e:
            content = f"【读取失败，错误信息】{str(e)}"

        # 写入头部信息
        out_f.write(f"【{idx}】文件路径：{file_path}\n")
        out_f.write("-" * 80 + "\n")
        out_f.write(content)
        out_f.write("\n")
        out_f.write("=" * 80 + "\n\n")

print(f"导出完成！")
print(f"共处理 {len(match_files)} 个代码文件")
print(f"完整内容文件保存路径：{os.path.abspath(OUTPUT_FILE)}")

