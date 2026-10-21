# Install

## Install

安装插件的方法：把仓库克隆到 AstrBot 的插件目录，然后重启宿主程序。
克隆完成后插件会被自动发现，无需手动注册。

```bash
cd /AstrBot/data/plugins
git clone https://github.com/wuyaomin248-stack/astrbot_plugin_rag_pro.git
```

## Embedding

启用向量检索前必须先配置 Embedding 模型。打开 AstrBot 的服务提供商设置，
添加任意 OpenAI 兼容的 embedding 模型，然后把插件配置里的
`embedding_provider_id` 填成该提供商 ID。留空时插件会自动选用第一个
已激活的 Embedding 提供商。

配置完成用 `/rag status` 验证，向量轨道应显示「正常」。
