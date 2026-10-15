# preprocess.py
import re


def clean_text(text):
    """
    清洗新闻组文本：
    1. 按第一个空行分离头部和正文
    2. 提取 Subject
    3. 去除引用行（以 > 开头）
    4. 去除 URL、邮箱
    5. 合并 Subject 和正文，去除多余空白
    """
    text = str(text)

    if '\n\n' in text:
        header, body = text.split('\n\n', 1)
    else:
        header, body = '', text

    subject = ''
    for line in header.split('\n'):
        if line.lower().startswith('subject:'):
            subject = line[len('subject:'):].strip()
            break

    lines = body.split('\n')
    clean_lines = [ln for ln in lines if not ln.strip().startswith('>')]
    body = ' '.join(clean_lines)

    combined = subject + ' ' + body

    combined = re.sub(r'http\S+|www\.\S+', ' ', combined)
    combined = re.sub(r'\S+@\S+', ' ', combined)
    combined = re.sub(r'\s+', ' ', combined).strip()

    return combined