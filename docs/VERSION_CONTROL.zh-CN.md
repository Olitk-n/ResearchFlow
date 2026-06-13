# Git 与 GitHub 版本管理指南

## 最简单的日常流程

每次开始工作前：

```powershell
git pull
```

完成一个小功能并测试通过后：

```powershell
git status
git add .
git commit -m "feat: 简短说明这次完成了什么"
git push
```

常用提交前缀：

- `feat:` 新功能
- `fix:` 修复问题
- `docs:` 修改文档
- `test:` 修改测试
- `chore:` 工具或配置调整

## 开发新功能

不要直接在 `main` 上做较大的修改。创建分支：

```powershell
git switch -c feature/功能英文名
```

开发完成后提交并上传：

```powershell
git add .
git commit -m "feat: 功能说明"
git push -u origin feature/功能英文名
```

然后在 GitHub 创建 Pull Request，确认自动测试通过后再合并。

## 发布版本

功能稳定时创建版本标签：

```powershell
git switch main
git pull
git tag -a v0.1.0 -m "ResearchFlow v0.1.0"
git push origin v0.1.0
```

版本号规则：修复问题增加最后一位，新功能增加中间一位，不兼容的大改增加第一位。

## 安全底线

- 永远不要提交 `.env`、API Key、密码、数据库和真实用户数据。
- 提交前先看 `git status` 和 `git diff --staged`。
- 不要随意使用 `git reset --hard`。
- 已经误传密钥时，仅删除文件不够，必须立即去供应商后台作废并更换密钥。
