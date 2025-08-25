# 使用一个官方的 Python 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 创建一个目录用于存放上传的文件
RUN mkdir -p /app/uploads

# 复制依赖文件并安装依赖
COPY requirements.txt requirements.txt
# PaddleOCR 和 PyTorch 很大，安装会很慢
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目所有文件到工作目录
COPY . .

# 暴露端口，需要和 Gunicorn 启动时使用的端口一致
EXPOSE 10000

# 定义容器启动时执行的命令
# 使用 Gunicorn 启动应用，它比 Flask 自带的服务器更稳定
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--bind", "0.0.0.0:10000", "app:app"]