# astrbot_plugin_rag_pro

> 基于 BM25 + 向量双轨混合召回与 RRF 融合的 AstrBot RAG 检索增强插件

## 这个插件解决什么问题

纯向量检索对**精确词**不敏感。用户问 `get_embeddings_batch` 或
`error_code_401` 这类标识符时，稠密向量把它们映射到语义空间里
一个模糊的位置，常常召回失败——而这恰恰是开发者提问的主要形式。

本插件在 AstrBot 的 `on_llm_request` 钩子上做非侵入式上下文注入：
每次请求前，用**稀疏 + 稠密**两条轨道并行召回，再用倒数排名融合
（RRF）合并，取 Top-K 注入到 system prompt 尾部。

## 架构

```text
用户提问
   │
   ├──────────────────────┐
   ▼                      ▼
[向量召回]              [BM25 倒排召回]
语义相似度             精确标识符 / 报错码
   │                      │
   └──────────┬───────────┘
              ▼
      [RRF 倒数排名融合]
       消除量纲差异，单轨可退化
              │
              ▼
      [相关性门槛]（可选，默认关闭）
              │
              ▼
      [Top-K 上下文预算装配]
       保证代码块闭合
              │
              ▼
      [on_llm_request 隐式注入]
```

## 目录结构

```text
astrbot_plugin_rag_pro/
├── main.py                       # 插件生命周期、on_llm_request 钩子、/rag 命令
├── metadata.yaml                 # 插件元信息
├── _conf_schema.json             # 控制台配置项定义
├── core/
│   ├── engine/
│   │   ├── bm25.py               # 零依赖 BM25 Okapi 实现
│   │   ├── rrf.py                # 倒数排名融合（含量纲归一化与单轨退化）
│   │   ├── vector.py             # 向量检索轨，对接宿主 Embedding Provider
│   │   ├── relevance.py          # IDF 加权查询相关性判定
│   │   └── kb_bridge.py          # 文档发现与原生知识库桥接
│   └── storage/
│       ├── splitter.py           # Markdown 结构感知切片（代码块保护）
│       └── index_db.py           # SQLite 持久化与 SHA-256 增量索引
├── eval/
│   ├── benchmark.json            # 20 条评测查询
│   ├── run_eval.py               # 离线评测脚本（Hit@1/5, MRR, NDCG@5）
│   └── docs/                     # 评测语料
└── tests/
    └── test_engine.py            # 单元测试
```

## 安装

```bash
cd /AstrBot/data/plugins
git clone https://github.com/wuyaomin248-stack/astrbot_plugin_rag_pro.git
```

重启 AstrBot，插件会被自动发现。

### 配置 Embedding 模型

向量轨需要一个 Embedding 提供商。在 AstrBot 的服务商设置里添加任意
OpenAI 兼容的 embedding 模型，然后把插件配置的 `embedding_provider_id`
填成该提供商 ID；留空则自动选用第一个已激活的。

未配置时插件**不会报错**，会退化到纯 BM25 单轨继续工作。

验证：

```
/rag status
```

## 评测结果

`eval/` 下的基准包含 20 条查询（18 条有标准答案 + 2 条离题干扰），
语料为 36 个切片。

```bash
python eval/run_eval.py --threshold 0.35
```

| 配置 | Hit@1 | Hit@5 | MRR | NDCG@5 |
|---|---|---|---|---|
| BM25 单轨 | 0.7222 | 0.8889 | 0.8056 | 0.8274 |
| **双轨 + RRF** | **0.7778** | **0.9444** | **0.8611** | **0.8614** |

双轨融合相对单轨：Hit@5 +5.6pt，MRR +5.6pt。提升主要来自
"语义改写提问"与"精确标识符提问"两类查询的互补。

```bash
# 换个配置对比
python eval/run_eval.py --config bm25_only --threshold 0.35
```

### 已知限制

离题查询（如 "how do I bake sourdough bread"）**仍会注入 Top-3 知识**。
原因是 RRF 只融合排名、丢弃绝对分数，任何命中一个常见词的查询都会
进入候选并被归一化成 0.5 分，高于默认阈值 0.35。

`min_query_coverage` 提供了一道 IDF 加权的相关性门槛来挡这种情况，
但在当前 36 切片的语料规模下，正常提问与离题提问的覆盖率区间重叠
（正常 0.09~0.59，离题 0.24~0.31），无法用单一阈值分开，因此**默认关闭**。
语料上到 100 切片以上后再启用，建议从 0.35 起试。

其它未实现项见 [Roadmap](#roadmap)。

## 日常使用

```
/rag status     # 查看索引状态、向量轨可用性、当前权重与阈值
/rag sync       # 增量同步（只处理新增/修改的文件）
/rag rebuild    # 清空索引并从磁盘全量重建
```

## 配置项

| 配置 | 默认 | 说明 |
|---|---|---|
| `enable_rag` | `true` | 总开关 |
| `enable_vector` | `true` | 启用向量轨；关闭则纯 BM25 |
| `embedding_provider_id` | `""` | 指定 Embedding 提供商，留空自动选第一个 |
| `score_threshold` | `0.35` | RRF 综合分截断。低于 0.5 = 单轨命中即注入 |
| `min_query_coverage` | `0.0` | 相关性门槛，0 为关闭（见已知限制） |
| `top_k` | `3` | 单次最多注入的切片数 |
| `max_context_chars` | `2500` | 注入内容的总字符预算 |
| `chunk_size` | `500` | 二级分块字符数 |
| `chunk_overlap` | `60` | 分块重叠字符数 |
| `bm25_weight` | `0.5` | RRF 中 BM25 轨权重 |
| `vector_weight` | `0.5` | RRF 中向量轨权重 |
| `suppress_native_rag` | `true` | 命中时压制原生知识库，避免双重注入 |
| `extra_kb_dirs` | 见 schema | 额外扫描的文档目录 |

## 关键设计说明

### 为什么用 RRF 而不是加权求和

BM25 分数无上界，余弦相似度在 [-1, 1]，两者不能直接相加。RRF 丢弃
原始分数、只融合排名位置，天然消除量纲问题。

融合后按理论最大值 `(bm25_weight + vector_weight) / (k + 1)` 归一化到
0~1，让阈值在不同权重配置下可比。参考 `core/engine/rrf.py`。

### 为什么中文用二元组

逐字切分会把「安装插件」拆成 安/装/插/件，这四个字在几乎任何中文
文档里都出现，等于没有信号。二元组保留词内相邻关系：
安装/装插/插件，不需要引入分词词典就能正常工作。

英文标识符则整段保留（`[a-zA-Z0-9_\-\.]+`），确保
`get_embeddings_batch` 不被切碎。

### 为什么要做增量索引

重算 embedding 是整个索引流程里最贵的一步。启动时先比 `mtime`，
变了再算 SHA-256，只有内容和时间戳都变的文件才重新向量化。
没有改动时重启零 embedding 调用。

### 为什么切片要保护代码块

被从中间截断的代码块会破坏注入内容的语法完整性，模型无法正确理解。
切片器把代码块当原子单元处理；超长代码块按行切分后重新包裹 ``` 围栏，
保证每个切片里的围栏都是配平的。

## 测试

```bash
python -m unittest discover -s tests -v    # 17 个单元测试
python eval/run_eval.py --threshold 0.35   # 检索质量评测
```

单元测试不需要 AstrBot 宿主，也不需要网络。

## Roadmap

以下能力在早期 README 里被描述过，但**当前代码并未实现**，列在这里
避免文档与实现不符：

- **Query Rewriter**：查询改写 / 同义词扩展 / 多路查询派生
- **Cross-Encoder Reranker**：RRF 之后再接一层精排模型
- **SimHash 去重**：切片级内容指纹去重
- **AST 语义切片**：目前只按 Markdown 标题层级切分，未解析代码 AST
- **CI**：单元测试与评测尚未接入 GitHub Actions
- **离题查询抑制**：见「已知限制」

## 许可

MIT，见 [LICENSE](LICENSE)。

## 作者

日海 & 橙子汐 ([@wuyaomin248-stack](https://github.com/wuyaomin248-stack))
