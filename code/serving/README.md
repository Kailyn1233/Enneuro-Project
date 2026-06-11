# EnNeuro Serving 模块使用文档

## 一、项目结构

```
serving/
├── __init__.py          # 模块导出
├── config.py            # 配置管理
├── logger.py            # 日志工具
├── schema.py            # 数据模型定义
├── predictor.py         # 本地预测器
├── server.py            # HTTP服务器
├── client.py            # HTTP客户端
├── metrics.py           # 指标监控
├── tcp_server.py        # TCP服务器
├── tcp_client.py        # TCP客户端
├── benchmark_client.py  # 压测客户端
└── tests/               # 测试文件
    ├── test_predictor.py
    ├── test_schema.py
    └── test_client_server.py
```

## 二、模块功能说明

| 文件 | 功能描述 |
|------|----------|
| `config.py` | 配置管理，支持环境变量配置（端口、模型路径、日志级别等） |
| `logger.py` | 日志工具，提供统一的日志接口 |
| `schema.py` | 数据模型，使用Pydantic定义请求/响应结构 |
| `predictor.py` | 本地预测器，加载模型并执行预测 |
| `server.py` | HTTP服务器，提供/ping、/health、/predict端点 |
| `client.py` | HTTP客户端，命令行工具调用服务 |
| `metrics.py` | 指标监控，收集请求计数、延迟等指标 |
| `tcp_server.py` | TCP协议服务器，支持低延迟通信 |
| `tcp_client.py` | TCP客户端，与TCP服务器通信 |
| `benchmark_client.py` | 压测工具，测试服务并发性能 |

## 三、启动服务

### 3.1 启动HTTP服务器

```bash
# 默认端口8080
python -m serving.server

# 自定义端口
python -c "from serving.server import ServingServer; server = ServingServer(port=8888); server.run()"
```

服务启动后访问：
- `http://localhost:8080/ping` - 健康检查
- `http://localhost:8080/health` - 服务状态
- `http://localhost:8080/predict` - 预测接口

### 3.2 启动TCP服务器

```bash
# 默认端口8081
python -m serving.tcp_server
```

## 四、客户端命令使用

### 4.1 HTTP客户端命令

```bash
# 1. 健康检查（ping）
python -m serving.client --action ping

# 2. 获取服务状态（health）
python -m serving.client --action health

# 3. 执行预测（predict）
python -m serving.client --action predict --input "[[1.0, 2.0, 3.0]]"

# 4. 指定服务器地址
python -m serving.client --host 127.0.0.1 --port 8080 --action ping
```

### 4.2 TCP客户端命令

```bash
# 1. TCP ping
python -m serving.tcp_client --action ping

# 2. TCP health检查
python -m serving.tcp_client --action health

# 3. TCP预测
python -m serving.tcp_client --action predict --input "[[1.0, 2.0, 3.0]]"
```

### 4.3 客户端参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 服务器地址 | localhost |
| `--port` | 服务器端口 | 8080 (HTTP) / 8081 (TCP) |
| `--action` | 操作类型 | 必填（ping/health/predict） |
| `--input` | 预测输入数据（JSON数组） | predict时必填 |

## 五、API接口说明

### 5.1 GET /ping

**功能**：简单健康检查

**请求**：无参数

**响应**：
```json
{
  "status": "success",
  "message": "pong"
}
```

### 5.2 GET /health

**功能**：获取服务详细状态

**请求**：无参数

**响应**：
```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "uptime_seconds": 120.5
}
```

### 5.3 POST /predict

**功能**：执行模型预测

**请求体**：
```json
{
  "inputs": [[1.0, 2.0, 3.0]],
  "version": "1.0"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| inputs | List[List[float]] | 是 | 输入数据，二维数组 |
| version | string | 否 | 模型版本 |

**响应**：
```json
{
  "predictions": [3.1, 3.8, 4.5],
  "model_version": "1.0.0",
  "latency_ms": 0.5
}
```

## 六、压测工具使用

### 6.1 运行压测

```bash
# 默认配置：10线程，每线程100请求
python -m serving.benchmark_client

# 自定义配置
python -m serving.benchmark_client --host localhost --port 8080 --threads 20 --requests 500

# 指定输入数据大小
python -m serving.benchmark_client --input-size 10
```

### 6.2 压测参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 服务器地址 | localhost |
| `--port` | 服务器端口 | 8080 |
| `--threads` | 并发线程数 | 10 |
| `--requests` | 每线程请求数 | 100 |
| `--input-size` | 输入向量维度 | 3 |

### 6.3 压测输出示例

```json
{
  "total_requests": 1000,
  "successful_requests": 1000,
  "failed_requests": 0,
  "success_rate": 100.0,
  "total_time_seconds": 2.35,
  "min_latency_ms": 0.1,
  "max_latency_ms": 5.2,
  "avg_latency_ms": 0.8,
  "p50_latency_ms": 0.5,
  "p90_latency_ms": 1.2,
  "p99_latency_ms": 3.5,
  "throughput": 425.5
}
```

## 七、测试命令

### 7.1 运行单元测试

```bash
# 测试预测器
python -m serving.tests.test_predictor -v

# 测试数据模型
python -m serving.tests.test_schema -v

# 测试客户端服务器集成
python -m serving.tests.test_client_server -v
```

### 7.2 手动测试

```bash
# 启动服务器（终端1）
python -m serving.server

# 在另一个终端测试
python -m serving.client --action ping
python -m serving.client --action health
python -m serving.client --action predict --input "[[1.0,2.0,3.0],[4.0,5.0,6.0]]"
```

## 八、配置说明

### 8.1 环境变量配置

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| SERVING_HOST | 服务绑定地址 | localhost |
| SERVING_PORT | HTTP服务端口 | 8080 |
| MODEL_PATH | 模型文件路径 | ./models |
| LOG_LEVEL | 日志级别 | INFO |
| METRICS_ENABLED | 是否启用指标 | true |

### 8.2 使用环境变量启动

```bash
# Linux/Mac
export SERVING_PORT=9090
export LOG_LEVEL=DEBUG
python -m serving.server

# Windows PowerShell
$env:SERVING_PORT=9090
$env:LOG_LEVEL="DEBUG"
python -m serving.server
```

## 九、完整工作流程示例

```bash
# 1. 启动服务
python -m serving.server

# 2. 检查服务是否正常
python -m serving.client --action ping
# 输出: {"status": "success", "message": "pong"}

# 3. 查看服务状态
python -m serving.client --action health
# 输出: {"status": "healthy", "model_version": "1.0.0", "uptime_seconds": 5.2}

# 4. 执行预测
python -m serving.client --action predict --input "[[1.0,2.0,3.0]]"
# 输出: {"predictions": [3.1, 3.8, 4.5], "model_version": "1.0.0", "latency_ms": 0.2}

# 5. 运行压测
python -m serving.benchmark_client --threads 20 --requests 200
# 输出压测报告
```

## 十、TCP vs HTTP 对比

| 特性 | HTTP | TCP |
|------|------|-----|
| 协议类型 | 应用层 | 传输层 |
| 延迟 | 较高 | 较低 |
| 适用场景 | 跨网络、浏览器访问 | 内网低延迟场景 |
| 端口 | 8080 | 8081 |
| 命令示例 | `serving.client` | `serving.tcp_client` |

## 十一、常见问题

### Q1: 服务启动失败？
- 检查端口是否被占用：`netstat -ano | findstr 8080`
- 尝试更换端口：`python -c "from serving.server import ServingServer; ServingServer(port=9090).run()"`

### Q2: 预测请求失败？
- 确保输入格式正确：`--input "[[1.0,2.0,3.0]]"`（注意是二维数组）
- 检查服务器是否正常运行

### Q3: 如何停止服务？
- 按 `Ctrl+C` 中断运行

### Q4: 如何查看日志？
- 服务器启动后自动输出日志到控制台
- 日志级别可通过 `LOG_LEVEL` 环境变量设置（DEBUG/INFO/WARNING/ERROR）
