# astrbot_plugin_rag_pro

> 基于 BM25 + 向量双轨混合召回与 Cross-Encoder 重排的 AstrBot 高级 RAG 检索增强插件

## 项目概述
本插件为 AstrBot 量身定制，采用非侵入式的 `on_llm_request` 上下文拦截挂载机制。针对纯向量检索对函数名、报错行号等精确词不敏感（召回率低）的痛点，引入关键词稀疏检索与 Cross-Encoder 语义重排，实现精准知识注入。

---

## 系统架构拓扑

```text
用户输入 Query
      │
      ▼ [1. Query Rewriter]
并发派生: 原查询 + 关键词抽取 + 扩展同义词 (2~3条)
      │
      ├──────────────────────┐
      ▼                      ▼
[2. 向量召回 (Dense)]    [3. BM25 倒排召回 (Sparse)]
语义相似度 Top-15         精确代码/报错码/术语 Top-15
      │                      │
      └──────────┬───────────┘
                 ▼
          [4. 倒数排名融合 (RRF)]
           初步合并去重，候选集 Top-20
                 │
                 ▼
          [5. Cross-Encoder Reranker]
           精准相关度二次打分 (过滤阈值 < 0.65)
                 │
                 ▼
          [6. 精选 Top 3~5 概念卡]
                 │
                 ▼
      [7. on_llm_request 隐式注入]
       装配至 System Prompt 尾部或倒数第 1 轮
```

---

## 目录结构规范

```text
astrbot_plugin_rag_pro/
├── main.py                     # 插件生命周期与 on_llm_request 钩子入口
├── _conf_schema.json           # 控制台配置（阈值、模型选择、Top-K）
├── core/
│   ├── preprocessor.py         # 文档切片、指纹去重（SimHash/MD5）与元数据提取
│   ├── query_rewriter.py       # 轻量大模型/规则实现的 Query 意图扩展与重写
│   ├── hybrid_retriever.py     # BM25 + 向量检索 双轨召回引擎
│   └── reranker.py             # Cross-Encoder / 本地轻量模型 重排序打分器
├── storage/
│   ├── bm25_index.py           # 本地 BM25 倒排索引（基于 SQLite 或 rank_bm25）
│   └── vector_store.py         # 向量数据库适配器（Chroma / Faiss）
└── eval/
    ├── benchmark.json          # 踩坑测试集（30条固定 Query + 期望命中 DocID）
    └── run_eval.py             # 离线自动化评测脚本（输出 NDCG@5、MRR）
```

---

## 落地演化路线 (4 阶段)

- **阶段一：入库预处理与索引清洗 (Day 1)**
  - 弃用机械定长切片，采用基于 Markdown 标题树与 AST 的语义块切片；
  - 引入 SimHash 内容指纹去重，保留最新版本；
  - 构建本地持久化 BM25 索引。
- **阶段二：双轨检索与 RRF 融合 (Day 2)**
  - 接入向量召回与 BM25 倒排并发查询；
  - 实现 Reciprocal Rank Fusion (RRF) 归一化融合算法。
- **阶段三：重排序与运行时注入 (Day 3)**
  - 接入 Cross-Encoder 重排模型，执行严格分数截断（Threshold < 0.65）；
  - 挂载 AstrBot `on_llm_request` Filter，在请求前动态注入精选 Top 3~5 概念卡。
- **阶段四：基准评测闭环 (Day 4)**
  - 构筑 30 道经典报错与高频提问基准集，对比验证召回率（Hit@1 / Hit@5）提升。

---
**架构设计**：日海 & 橙子汐 (@wuyaomin248-stack)
