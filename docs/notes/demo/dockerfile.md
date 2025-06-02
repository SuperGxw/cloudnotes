---
title: Dockerfile构建命令以及docker容器常用命令
createTime: 2024/09/28 16:40:29
permalink: /article/zdhr4ufp/
---
## Dockerfile命令
在 Docker 中，常使用 Dockerfile 来定义如何构建自己的 Docker 镜像。下述是一些常用的 Dockerfile 命令：
- FROM：指定基础镜像
- LABEL：为镜像添加元数据标签
- RUN：在镜像中执行命令
- COPY：将文件或目录复制到镜像中
- ADD：将文件添加到镜像中
- ENV：设置环境变量
- EXPOSE：指定运行容器时应监听的端口
- ENTRYPOINT：定义容器启动时运行的命令
- CMD：为容器提供默认运行命令
- VOLUME：创建可被容器挂载的目录
- USER：指定运行容器时的用户名或UID
- WORKDIR：设置工作目录
- ARG：定义一个变量，在docker build 时可以用 --build-arg 来赋值
- ONBUILD：当一个被继承的Dockerfile中，将这个指令指定的命令作为指令执行
- STOPSIGNAL：设置停止信号
- HEALTHCHECK：设置健康检查
- SHELL：设置SHELL命令
- CREATED：设置构建时间
## Docker命令示例
Docker 是一种流行的容器化平台，下述是一些常用的 Docker 命令：

```shell
# 构建镜像 （注意末尾空格+.）
docker build -f Dockerfile -t [image_name] .

# 创建并运行容器，
# -it/d 用于指定容器的运行模式, -it: 以交互模式运行容器; -d: 以守护进程（后台）模式运行容器
# -p 在运行容器时将容器的端口映射到主机上
# --name 用于指定容器的名称
# -v 挂载本地目录到 Docker 容器
# -rm 在退出 bash 会话后，Docker 会自动删除这个容器。
docker run -it/-d -p 8888:8888 --name [container_name] -v /path/to/local/dir:/app/data [image_name]

# 启动 Jupyter Lab 时，使用 --ip=0.0.0.0 参数，使其绑定到所有接口，从而允许从外部访问
jupyter lab --ip=0.0.0.0 --no-browser --allow-root --port 8888

# 查看本机已有镜像信息
docker images

# 查看容器
docker ps

# 删除镜像
docker rmi -f [image_name]/[image_id]

# 删除容器
docker rm -f [container_name]/[container_id]

# 在已运行的容器中执行命令，例如执行 Bash shell
# exec: 在容器中执行新的命令，不会影响容器的当前会话
# attach: 附加到容器的会话，可以与容器的当前会话进行交互
docker exec/attach [container_name]/[container_id] bin/bash

# 启动一个已停止的容器
docker start [container_name]/[container_id]

# 关闭正在运行的容器
docker stop [container_name]/[container_id]
```