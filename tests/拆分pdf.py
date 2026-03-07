import fitz  # PyMuPDF的核心库


def split_pdf(input_path, output_path, start_page, end_page):
    """
    拆分PDF指定页码范围的页面
    :param input_path: 原始PDF文件路径（如："论文.pdf"）
    :param output_path: 拆分后保存的PDF路径（如："论文_24-50页.pdf"）
    :param start_page: 起始页码（文档页码，从1开始）
    :param end_page: 结束页码（文档页码，从1开始）
    """
    # 打开原始PDF
    doc = fitz.open(input_path)

    # 验证页码范围合法性
    total_pages = doc.page_count
    if start_page < 1 or end_page > total_pages or start_page > end_page:
        print(f"页码范围无效！总页数：{total_pages}，请输入1-{total_pages}内的有效范围")
        return

    # PyMuPDF页码从0开始，需转换为索引（文档页码-1）
    start_idx = start_page - 1
    end_idx = end_page - 1

    # 创建新的PDF文档
    new_doc = fitz.open()
    # 插入指定范围的页面
    new_doc.insert_pdf(doc, from_page=start_idx, to_page=end_idx)

    # 保存拆分后的PDF
    new_doc.save(output_path)
    new_doc.close()
    doc.close()
    print(f"PDF拆分完成！已保存到：{output_path}")


# -------------------------- 自定义参数（修改这里即可）--------------------------
INPUT_PDF = "D:\whale\文献\AAA importance\面向多目标车间动静态设施布局优化的构形空间进化算法_刘思妤.pdf"  # 原始PDF文件路径
OUTPUT_PDF = "论文_24-50页.pdf"  # 拆分后保存的文件名
START_PAGE = 24  # 起始页码
END_PAGE = 50  # 结束页码
# -----------------------------------------------------------------------------

# 执行拆分
if __name__ == "__main__":
    split_pdf(INPUT_PDF, OUTPUT_PDF, START_PAGE, END_PAGE)