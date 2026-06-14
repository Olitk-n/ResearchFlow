# ResearchFlow 小白使用指南

## 1. 第一次安装

需要安装：

- Node.js 20 或更高版本
- uv
- Docker Desktop（只有运行自动生成的实验代码时需要）

在项目文件夹空白处按住 `Shift` 并单击鼠标右键，选择“在终端中打开”，依次运行：

```powershell
Copy-Item .env.example .env
uv sync --directory apps/api
npm install --prefix apps/web
```

打开 `.env`，把 `SECRET_KEY` 和 `ENCRYPTION_KEY` 后面的示例文字替换为两个不同的长随机字符串。`.env` 包含密码和 API Key，不要发给别人，也不要上传到 GitHub。

## 2. 每次启动和停止

先确认 Docker Desktop 已经显示“Engine running”，然后在项目终端运行：

```powershell
.\scripts\start-local.ps1
```

浏览器打开：

- 操作界面：<http://localhost:3000>
- 后端接口说明：查看启动窗口显示的 `API` 地址。默认是
  <http://127.0.0.1:8000/docs>，当前 `.env` 如果配置 `API_PORT=8002`，则应打开
  <http://127.0.0.1:8002/docs>。

启动脚本会先生成稳定的生产版本，再自动检查网页样式和后端。停止系统：

```powershell
.\scripts\stop-local.ps1
```

如果 `.env` 中的 `API_PORT` 不是 `8000`，接口说明地址也要改成对应端口。

## 3. 实际使用流程

1. 注册本地账号并登录。
2. 在“模型设置”添加供应商、模型名、API 地址和 API Key。
3. 先运行连接测试，确认模型状态为可用。
4. 新建科研项目，输入研究方向，例如“LLM 智能体评测”。
5. 启动文献检索，查看论文、证据和低覆盖研究空白候选。
6. 人工选择一个候选方向，再确认数据集许可和实验成本。
7. 生成实验并在 Docker 沙箱中运行。
8. 检查真实实验结果和引用来源。
9. 选择 ICLR、ICML、NeurIPS 或 arXiv 模板，生成 LaTeX、BibTeX 和 PDF。

“低覆盖研究空白”不是“世界上绝对没人发表过”。正式投稿前仍应人工复查检索结果、实验设计和论文内容。

## 4. 常见问题

### 浏览器显示拒绝连接

说明程序没有启动成功。重新运行启动脚本，并保持 Docker Desktop开启。

### 实验显示无法连接 Docker API

启动 Docker Desktop，等到它显示运行完成，再重新执行实验。

### 修改了代码但页面没变化

先停止系统，再重新启动。仍无变化时按 `Ctrl + F5` 强制刷新浏览器。

### 如何检查项目是否正常

```powershell
.\scripts\test-local.ps1
```

测试全部通过后再提交版本。
