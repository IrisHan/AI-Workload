import type { OpNode, Template } from "./analysis";

interface Meaning { kind: string; description: string; memory: string }
const viewMemory = "逻辑张量仍存在；reshape/transpose 可能共享输入存储。是否发生复制取决于布局和运行时，不代表新增同等显存。";
const computeMemory = "无独立权重不等于无计算或无激活。实际分配与算子融合、执行阶段及张量生命周期有关。";

export function nodeMeaning(id: string): Meaning | undefined {
  const name = id.startsWith("block.") ? id.slice(6) : id;
  if (["q_heads", "k_heads", "v_heads"].includes(name))
    return { kind: "张量视图", description: "将投影输出拆分为多个注意力头，不新增权重；投影参数属于上游模块。", memory: viewMemory };
  if (["concat", "kv_split", "k_split", "v_split"].includes(name))
    return { kind: "张量布局", description: "拆分或合并张量维度，不新增可学习权重。", memory: viewMemory };
  if (["input", "output"].includes(name))
    return { kind: "张量边界", description: "层输入或输出的激活张量，不是权重矩阵。", memory: computeMemory };
  if (["rope_q", "rope_k"].includes(name))
    return { kind: "位置旋转", description: "按位置旋转 Q/K；当前模型的 RoPE 不含可学习权重。旋转缓存不属于权重参数。", memory: computeMemory };
  if (["qk_matmul", "av_matmul"].includes(name))
    return { kind: "激活矩阵乘", description: "矩阵乘的操作数来自激活张量，不是新增权重。FLOPs 非零；融合 attention 可避免完整分数矩阵落显存。", memory: computeMemory };
  if (name === "softmax")
    return { kind: "归一化运算", description: "对注意力分数做 softmax，不持有可学习权重。", memory: computeMemory };
  if (name === "scale_mask")
    return { kind: "缩放与掩码", description: "缩放注意力分数并应用掩码，不新增权重。", memory: computeMemory };
  if (["attn_add", "ffn_add", "expert_sum"].includes(name))
    return { kind: "张量求和", description: "对已有激活做残差或专家加权求和，权重参数不在此重复计数。", memory: computeMemory };
  if (name === "silu" || name.endsWith(".silu"))
    return { kind: "激活函数", description: "SiLU 逐元素非线性变换；激活函数不是可学习参数。", memory: computeMemory };
  if (name === "multiply" || name.endsWith(".multiply"))
    return { kind: "逐元素乘", description: "对门控和上投影的激活逐元素相乘，不新增权重。", memory: computeMemory };
  if (name === "top_k")
    return { kind: "路由选择", description: "选择专家；可学习路由权重计入 router，而非 top-k 选择操作。", memory: computeMemory };
  if (["partial_rope", "inverse_rope", "compressed_rope"].includes(name))
    return { kind: "位置旋转", description: "对指定通道应用位置旋转或逆旋转，无独立权重；压缩器的位置参数在 ape 模块单独核算。", memory: computeMemory };
  if (/sinkhorn|hc_.*_(pre|post|collapse)/.test(name))
    return { kind: "残差混合", description: "使用已核算的 mHC 投影、base 和 scale 做归一化与残差混合，不重复计权重。", memory: computeMemory };
  if (/clamped_swiglu|silu_multiply|l2_normalize|output_gate|router_scoring|decay_gate|beta_gate/.test(name))
    return { kind: "逐元素运算", description: "激活、裁剪、归一化或门控运算，无新增独立权重。", memory: computeMemory };
  if (/causal_window|mask_softmax_sink|causal_attention/.test(name))
    return { kind: "注意力运算", description: "应用掩码、归一化或注意力加权；sink 权重由独立模块核算，缓存和激活不是权重。", memory: computeMemory };
  if (/sparse_select|expert_selection|indexer.top_k|hash_lookup/.test(name))
    return { kind: "索引选择", description: "选择 token 或专家；整数路由表和缓存单列，不计入浮点权重参数。", memory: computeMemory };
  if (/indexer.score|\.pool$|weighted_expert_sum|residual_add|output_add|gated_delta_recurrence/.test(name))
    return { kind: "张量计算", description: "基于输入、状态及已核算模块执行张量计算；此计算节点不新增独立权重。", memory: computeMemory };
  if (["token_ids", "image", "token_embeddings", "hidden_input", "language_embeddings"].includes(name))
    return { kind: "张量边界", description: "运行时输入或中间张量，不是独立权重矩阵。", memory: computeMemory };
  return undefined;
}

export function NodeDetails({ node, template, onSelect }: {
  node: OpNode; template: Template; onSelect: (id: string) => void;
}) {
  const meaning = nodeMeaning(node.id);
  const inputs = template.edges.filter(edge => edge.target === node.id)
    .map(edge => template.nodes.find(item => item.id === edge.source))
    .filter((item): item is OpNode => !!item);
  const outputs = template.edges.filter(edge => edge.source === node.id)
    .map(edge => template.nodes.find(item => item.id === edge.target))
    .filter((item): item is OpNode => !!item);
  const name = node.id.startsWith("block.") ? node.id.slice(6) : node.id;
  const heads = name === "q_heads" ? template.backbone?.num_heads : template.backbone?.num_kv_heads;
  const headDim = template.backbone?.head_dim;
  const standardHeads = ["q_heads", "k_heads", "v_heads"].includes(name)
    && ["MHA", "GQA", "MQA", "SWA"].includes(template.attn_kind) && heads && headDim;
  return <div className="detail-content" data-testid="operator-detail">
    <span className="category-label">{meaning?.kind ?? node.category}</span><h3>{node.label}</h3>
    <p>{meaning?.description ?? node.detail ?? "该快照未提供算子语义。"}</p>
    <dl className="operator-facts">
      <dt>独立权重参数</dt><dd>{meaning ? "0 · 无独立权重" : "未提供独立参数账本"}</dd>
      <dt>输入节点</dt><dd className="node-links">{inputs.length ? inputs.map(item =>
        <button className="text-button" key={item.id} onClick={() => onSelect(item.id)}>{item.label}</button>) : "层输入"}</dd>
      {inputs.map(item => {
        const module = template.modules.find(module => module.id === item.module_id);
        return module?.output_shape ? <dd key={item.id} className="mono">{item.label}: {module.output_shape}</dd> : null;
      })}
      {standardHeads ? <><dt>输出激活形状（逻辑）</dt><dd className="mono">[B, S, {heads}, {headDim}]</dd>
        <dt>Attention 布局（转置后）</dt><dd className="mono">[B, {heads}, S, {headDim}]</dd></> : null}
      <dt>输出节点</dt><dd className="node-links">{outputs.length ? outputs.map(item =>
        <button className="text-button" key={item.id} onClick={() => onSelect(item.id)}>{item.label}</button>) : "层输出"}</dd>
      <dt>激活张量大小</dt><dd>元素数 × dtype 字节数；B 为 batch，S 为本次处理的 token 数。当前结构快照未指定运行场景。</dd>
      <dt>新增分配 / 峰值显存</dt><dd>未评估，不能由参数量推断。</dd>
    </dl>
    {meaning && <p className="module-notes">{meaning.memory}</p>}
  </div>;
}
