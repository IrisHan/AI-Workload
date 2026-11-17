import { useId } from "react";

const descriptions: Record<string, string> = {
  "Mode": "Online按请求到达流仿真；Offline处理固定数量的请求。",
  "Arrival": "均匀到达采用固定间隔；Poisson采用指数间隔，由Seed固定随机序列。",
  "Continuous Batching": "开启后已完成请求可释放槽位并接纳新请求；关闭后按完整批次执行。",
  "Topology": "节点内外的互连情景。默认值来自所选硬件，覆盖值作用于本次设计。",
  "Search Mode": "Auto搜索合法并行组合；Manual计算指定的TP/PP/EP/DP方案。",
  "Scheduler": "Continuous与Sequential选择调度方式，并同步Continuous Batching字段。",
  "Weight Precision": "驻留模型权重的字节精度；只支持已核验的BF16和FP32。",
  "Activation Precision": "激活字节精度及矩阵算力档位；标量算子仍使用FP32算力。",
  "KV Precision": "K/V缓存的存储精度，参与每页字节和访存耗时计算。",
  "SLA Policy": "Soft保留未达标方案供比较；Hard仅接受仿真已证实满足延迟目标的方案。",
  "Prompt Tokens": "每个请求的输入长度。与生成长度共同决定KV缓存和上下文上限。",
  "Output Tokens": "每个请求生成的整数token数。首token由prefill产生，之后执行decode。",
  "Batch Size": "每个Attention组每步最多处理的请求数，不是总请求数。",
  "Concurrency": "整个部署允许同时驻留的请求数，按请求组分配，用于KV容量准入。",
  "Request Rate (req/s)": "Online每秒到达的请求数；超过处理能力时会产生排队。",
  "Request Count": "Offline任务需要完成的请求总数。",
  "Prefill Chunk": "每个请求一次prefill最多处理的token数，不能超过Batch tokens预算。",
  "Duration (s)": "仿真观察窗口的终止时间；Online在窗口结束时保留未完成请求的删失信息。",
  "Warmup (s)": "从指标统计中排除的初始预热时长，必须小于Duration。",
  "Seed": "Poisson随机到达的种子。输入和种子相同可复现；均匀到达不依赖它。",
  "TP": "Tensor Parallel。每层Attention及Dense张量按rank切分；不是把模型层分成几组。",
  "PP": "Pipeline Parallel。按连续层区间分stage，embedding在首stage、head在末stage。",
  "DP": "Data Parallel。Dense为完整请求副本；vLLM MoE的专家可能跨DP组分片。",
  "EP": "Expert Parallel。分配专家所有权；框架模式中复用TP或TP×DP设备，不额外乘卡数。",
  "PP Boundaries": "零基层边界，例如28层填写10表示[0,10)和[10,28)。留空采用框架默认分配。",
  "Memory Efficiency": "非权重池的有效利用率。额外碎片=(KV+Activation+Comm)×(1/eff−1)。",
  "Activation Efficiency": "激活池有效率。预算为必要激活字节/eff；1表示没有额外池开销，可改为小于1。",
  "KV Cache Efficiency": "KV池有效率。预算为分页后的KV字节/eff；尾页浪费已包含，不重复估计。",
  "Safety Margin": "保留的HBM容量比例，独立于碎片和运行时固定开销。",
  "Compute Efficiency": "矩阵/标量算子的有效算力比例。算力耗时=工作量/(核验峰值×eff)。",
  "HBM Efficiency": "有效HBM带宽比例。访存耗时=traffic/(Memory BW×eff)。",
  "Draft Tokens": "每次投机cycle提出的串行draft token数K，目标模型验证K+1个位置。",
  "Acceptance Rate": "每个draft在前序均接受时的条件接受概率a；这是情景假设，不是框架CLI。",
  "Expected Accepted Tokens": "每cycle接受的draft期望值Σaⁱ，不含额外输出token，由K和a自动推导。",
  "Expected Emitted Tokens": "包含额外token的期望输出：1+Σaⁱ。Excel R6使用这一口径。",
  "Draft Latency (ms)": "每cycle生成draft及相关开销的显式假设。空值表示未知；0是乐观下界。",
  "Draft Memory (GB)": "每GPU额外draft权重、缓存和工作区总预算。SI GB；空值表示未知。",
  "Intra Injection (GB/s)": "单GPU向节点内互连的注入带宽；双向NVLink总规格不能直接当单向带宽。",
  "Inter / GPU (GB/s)": "每GPU可使用的节点间单向网络带宽，仍受共享NIC限制。",
  "NIC Count": "每节点物理NIC数量；需要与NIC Port带宽成对设置。",
  "NIC Port (GB/s)": "单个NIC端口单向带宽。节点网络上限=数量×端口带宽/超售比。",
  "Nodes": "可供此部署使用的节点数量，与GPUs / Node共同形成资源预算。",
  "GPUs / Node": "每节点GPU预算；SXM模板不超过Hardware Center核验节点规格。",
  "Batch tokens": "每个Attention组每步的token总预算，限制prefill、decode和验证批次。",
  "Fairness Interval": "连续decode的公平调度轮次，达到间隔时给prefill保留机会。",
  "runtime_fixed_bytes": "每GPU固定runtime/context开销，整数bytes；可按显式情景覆盖。",
  "comm_reserve_bytes": "每GPU通信池预留bytes，不能低于必要通信缓冲下界。",
  "launch_time_us": "每个算子固定启动开销，微秒；属于情景假设。",
  "routing_imbalance": "专家路由热点造成的时间压力系数，不用于虚增总FLOPs。",
  "hbm_bandwidth_gb_s": "显式HBM带宽情景，SI GB/s；0采用Hardware Center来源，不改写来源记录。",
  "oversubscription": "共享网络超售系数，至少为1；越大可用带宽越小。",
  "link_efficiency": "互连有效带宽比例，与HBM效率独立。",
  "intra_latency_us": "节点内链路启动延迟，微秒。",
  "inter_latency_us": "节点间链路启动延迟，微秒。",
  "ttft_ms": "首token端到端延迟目标，包含排队，毫秒；留空不设置目标。",
  "itl_ms": "相邻输出token间隔目标，毫秒；留空不设置目标。",
  "top_k": "保留的候选方案数量，不是MoE每token选择的专家数。",
  "max_candidates": "单次搜索最多检查的候选数量，达到上限报告部分覆盖。",
  "max_seconds": "候选搜索墙钟时间预算，秒。",
  "event_budget": "调度仿真的事件预算；超出后报告截断，不能冒充完整结果。",
};

export function FieldName({ name, hint }: { name: string; hint?: string }) {
  const id = useId();
  const text = hint ?? descriptions[name] ?? (name.startsWith("Allowed ")
    ? `Auto搜索的${name.slice(8)}允许集合，逗号分隔。留空恢复来源和框架派生范围。` : "");
  return <span className={text ? "wc-field-name" : ""} tabIndex={text ? 0 : undefined}
    aria-describedby={text ? id : undefined}>{name}{text && <span id={id} role="tooltip">{text}</span>}</span>;
}

export function Readout({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return <div className="wc-readout"><FieldName name={label} hint={hint} /><output>{value}</output></div>;
}
