from PyPDF2 import PdfMerger
import os

def merge_two_pdfs(pdf1_path, pdf2_path, output_path):
    """
    合并两个PDF文件
    :param pdf1_path: 第一个PDF文件路径
    :param pdf2_path: 第二个PDF文件路径
    :param output_path: 合并后输出的文件路径
    """
    # 初始化合并器
    merger = PdfMerger()

    try:
        # 检查文件是否存在
        if not os.path.exists(pdf1_path):
            raise FileNotFoundError(f"未找到文件: {pdf1_path}")
        if not os.path.exists(pdf2_path):
            raise FileNotFoundError(f"未找到文件: {pdf2_path}")

        # 添加PDF文件
        merger.append(pdf1_path)
        merger.append(pdf2_path)

        # 写入输出文件
        merger.write(output_path)
        print(f"合并成功！文件已保存至: {output_path}")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 关闭合并器
        merger.close()

# --- 使用示例 ---
if __name__ == "__main__":
    # 在这里修改你的文件路径
    pdf_file_1 = r"D:\whale\入党\梁韵贤志愿服务小结.pdf"  # 第一个PDF
    pdf_file_2 = r"D:\whale\入党\2025-26学年党建学生组织论坛志愿者志愿时说明.pdf"  # 第二个PDF
    output_file = "D:\whale\入党\merged_result.pdf" # 合并后的文件名

    merge_two_pdfs(pdf_file_1, pdf_file_2, output_file)