# -*- coding: utf-8 -*-
"""
可复现的报告构建脚本
======================

将 report/机器学习优化器比较实验报告.md 转换为最终 PDF。

流程：Markdown → HTML（含 CSS 分页、页眉页脚、公式）→ headless Chrome 打印 PDF。

用法：
    python scripts/build_report.py

依赖：
    - Python 标准库 + markdown
    - Google Chrome / Microsoft Edge（用于打印 PDF）
"""

import datetime
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_MD = os.path.join(ROOT, "report", "机器学习优化器比较实验报告.md")
REPORT_HTML = os.path.join(ROOT, "report", "机器学习优化器比较实验报告.html")
REPORT_PDF = os.path.join(ROOT, "report", "机器学习优化器比较实验报告.pdf")

try:
    import markdown as md
except ImportError:
    print("缺少 markdown 库，请先安装：pip install markdown")
    sys.exit(1)


def find_browser():
    """查找 Chrome / Edge 可执行文件。"""
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # 尝试从 PATH 找
    for name in ["chrome", "google-chrome", "msedge", "chromium"]:
        p = shutil.which(name)
        if p:
            return p
    return None


CSS = """
@page {
    size: A4;
    margin: 2.0cm 1.8cm 2.2cm 1.8cm;
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
        color: #555;
    }
}

body {
    font-family: "Microsoft YaHei", "SimSun", "PingFang SC", sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #222;
    max-width: 100%;
}

h1 {
    font-size: 20pt;
    text-align: center;
    margin: 0.5em 0 0.8em;
}

h2 {
    font-size: 15pt;
    border-bottom: 1.5px solid #333;
    padding-bottom: 0.2em;
    margin-top: 1.3em;
    page-break-after: avoid;
}

h3 {
    font-size: 13pt;
    margin-top: 1.1em;
    page-break-after: avoid;
}

h4 {
    font-size: 11.5pt;
    page-break-after: avoid;
}

p {
    margin: 0.4em 0;
    text-align: justify;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.8em 0;
    font-size: 10pt;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #999;
    padding: 5px 8px;
    text-align: center;
}

th {
    background: #f0f0f0;
    font-weight: bold;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0.8em auto;
    page-break-inside: avoid;
}

code {
    font-family: Consolas, "Courier New", monospace;
    font-size: 9.5pt;
    background: #f5f5f5;
    padding: 1px 4px;
    border-radius: 3px;
}

pre {
    background: #f5f5f5;
    border: 1px solid #ddd;
    padding: 10px;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 9pt;
    page-break-inside: avoid;
}

pre code {
    background: none;
    padding: 0;
}

blockquote {
    border-left: 4px solid #ccc;
    margin: 0.6em 0;
    padding: 0.3em 1em;
    color: #555;
    background: #fafafa;
}

.formula {
    text-align: center;
    margin: 0.7em 0;
}

.formula img {
    display: block;
    margin: 0.3em auto;
    max-height: 3.5em;
}

.inline-formula {
    vertical-align: middle;
}

/* 封面 */
.cover {
    text-align: center;
    margin-top: 30%;
}
.cover h1 { font-size: 26pt; margin-bottom: 0.5em; }
.cover .title { font-size: 18pt; font-weight: bold; margin: 0.6em 0; }
.cover .info { font-size: 14pt; margin: 0.4em 0; }

.page-break { page-break-before: always; }
"""


def render_formula_png(latex, path, fontsize=13):
    """用 matplotlib mathtext 将 LaTeX 公式渲染为透明背景 PNG。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=fontsize)
    fig.savefig(path, dpi=300, bbox_inches="tight",
                pad_inches=0.03, transparent=True)
    plt.close(fig)


def build_html(md_text, cover_html, fig_dir):
    """将 Markdown 转为 HTML，套用 CSS，注入封面，公式渲染为图片。"""
    import re
    import hashlib

    os.makedirs(fig_dir, exist_ok=True)

    # ---- 第 1 步：在 markdown 转换【之前】提取公式，避免下划线被当作斜体 ----
    formula_store = {}  # placeholder -> (latex, is_block)

    def stash(m):
        latex = (m.group(1) or m.group(2)).strip()
        is_block = m.group(1) is not None
        key = f"FORMULAPH{len(formula_store)}X"
        formula_store[key] = (latex, is_block)
        return key

    # $$block$$ 优先于 $inline$
    md_text = re.sub(r"\$\$(.+?)\$\$|\$([^$\n]+?)\$", stash, md_text, flags=re.DOTALL)

    # ---- 第 2 步：markdown 转换 ----
    body = md.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )

    # ---- 第 3 步：渲染公式并替换占位符 ----
    def render(latex, fontsize):
        fname = f"formula_{hashlib.md5(latex.encode()).hexdigest()[:10]}.png"
        fpath = os.path.join(fig_dir, fname)
        if not os.path.isfile(fpath):
            render_formula_png(latex, fpath, fontsize=fontsize)
        return fname

    for key, (latex, is_block) in formula_store.items():
        try:
            fname = render(latex, fontsize=13 if is_block else 11)
            style = 'style="max-height:3.5em"' if is_block else 'style="max-height:1.3em" class="inline-formula"'
            html_snippet = f'<img src="figures/{fname}" alt="formula" {style}>'
            if is_block:
                html_snippet = f'<div class="formula">{html_snippet}</div>'
        except Exception as e:
            print(f"[WARN] 公式渲染失败: {latex[:40]}... ({e})")
            html_snippet = f"<code>{latex}</code>"
            if is_block:
                html_snippet = f'<div class="formula">{html_snippet}</div>'
        body = body.replace(key, html_snippet)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>机器学习优化器比较实验报告</title>
<style>{CSS}</style>
</head>
<body>
{cover_html}
<div class="page-break"></div>
{body}
</body>
</html>"""
    return html


def build_cover():
    today = datetime.date.today().strftime("%Y-%m-%d")
    return f"""
<div class="cover">
    <h1>机器学习课程实验报告</h1>
    <div class="title">实验题目：基于 Fashion-MNIST 的 MLP 图像分类及优化器性能比较</div>
    <div class="info">学生：汤鑫</div>
    <div class="info">日期：{today}</div>
</div>
"""


def main():
    print("=" * 60)
    print("构建实验报告 PDF")
    print("=" * 60)

    if not os.path.isfile(REPORT_MD):
        print(f"[FAIL] 找不到报告源文件: {REPORT_MD}")
        sys.exit(1)

    with open(REPORT_MD, encoding="utf-8") as f:
        md_text = f.read()

    # 剥离封面信息（源文件中的封面字段）——封面单独生成
    cover_html = build_cover()

    # 公式图片目录（与 HTML 同级的 figures 子目录）
    formula_dir = os.path.join(ROOT, "report", "figures")
    html = build_html(md_text, cover_html, formula_dir)
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] 已生成 HTML: {os.path.basename(REPORT_HTML)}")

    # 用 headless Chrome 打印 PDF
    browser = find_browser()
    if browser is None:
        print("[FAIL] 未找到 Chrome/Edge，无法打印 PDF")
        sys.exit(1)

    print(f"[OK] 使用浏览器: {browser}")
    file_url = "file:///" + REPORT_HTML.replace("\\", "/")

    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={REPORT_PDF}",
        "--print-to-pdf-no-header",
        file_url,
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)

    if os.path.isfile(REPORT_PDF):
        size = os.path.getsize(REPORT_PDF)
        print(f"[OK] 已生成 PDF: {os.path.basename(REPORT_PDF)} ({size} bytes)")
    else:
        print("[FAIL] PDF 生成失败")
        sys.exit(1)

    print("\n构建完成。")


if __name__ == "__main__":
    main()
