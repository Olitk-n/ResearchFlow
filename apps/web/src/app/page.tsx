"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  BookOpen,
  Box,
  BrainCircuit,
  Check,
  ChevronRight,
  CircleDot,
  Code2,
  Database,
  Download,
  ExternalLink,
  FileText,
  FlaskConical,
  KeyRound,
  Layers3,
  LoaderCircle,
  LockKeyhole,
  LogOut,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  Target,
  X,
  Zap,
} from "@/components/icons";
import {
  ApiError,
  DataPreparation,
  Dataset,
  Experiment,
  ExperimentRun,
  Gap,
  GapValidation,
  Paper,
  Project,
  ProjectDetail,
  api,
  downloadArtifact,
  subscribeProjectEvents,
} from "@/lib/api";

type View = "overview" | "literature" | "gaps" | "data" | "experiment" | "paper";
const QUALITY_LABELS: Record<string, string> = {
  concept_draft: "概念草稿",
  synthetic_demonstration: "模拟演示",
  initial_experiment: "初步实验",
  reproducible_research: "可复现研究",
  submission_candidate: "投稿候选",
};
type ModelConfig = {
  id: string;
  name: string;
  provider: string;
  model: string;
  key_hint: string;
  budget_limit_usd: number;
  spent_usd: number;
  remaining_budget_usd: number;
  input_price_per_million_usd?: number;
  output_price_per_million_usd?: number;
  is_default: boolean;
  capabilities: {
    reachable?: boolean;
    structured_output?: boolean;
    error?: string;
  };
};
const providerDefaultModels: Record<string, string> = {
  deepseek: "deepseek-chat",
  mimo: "mimo-v2.5-pro",
  openai: "gpt-4.1-mini",
  anthropic: "claude-sonnet-4-5",
  gemini: "gemini-2.5-flash",
  openrouter: "openai/gpt-4.1-mini",
  qwen: "qwen-plus",
  kimi: "kimi-k2.5",
  glm: "glm-4.5",
  minimax: "MiniMax-M2.1",
  openai_compatible: "",
};
type Readiness = {
  software_ready: boolean;
  acceptance_ready: boolean;
  checks: Record<string, boolean>;
  next_action?: string;
};

const statusLabel: Record<string, string> = {
  draft: "等待启动",
  discovering: "检索前沿",
  awaiting_topic: "等待选题",
  planning: "准备实验",
  paused: "已暂停",
  ready: "实验就绪",
  failed: "需要检查",
};

const navItems: Array<{ id: View; label: string; icon: typeof Layers3 }> = [
  { id: "overview", label: "研究总览", icon: Layers3 },
  { id: "literature", label: "前沿论文", icon: BookOpen },
  { id: "gaps", label: "研究空白", icon: Target },
  { id: "data", label: "数据集", icon: Database },
  { id: "experiment", label: "实验工作台", icon: FlaskConical },
  { id: "paper", label: "论文产物", icon: FileText },
];

function Logo() {
  return (
    <div className="logo">
      <div className="logo-mark"><BrainCircuit size={22} /></div>
      <div><strong>ResearchFlow</strong><span>科研自动化工作台</span></div>
    </div>
  );
}

function AuthScreen({ onAuthenticated }: { onAuthenticated: (token: string) => void }) {
  const [mode, setMode] = useState<"register" | "login">("register");
  const [email, setEmail] = useState("researcher@local.dev");
  const [password, setPassword] = useState("researchflow");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const result = await api<{ access_token: string }>(`/auth/${mode}`, {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      localStorage.setItem("researchflow_token", result.access_token);
      onAuthenticated(result.access_token);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法登录");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-shell">
      <section className="auth-story">
        <Logo />
        <div className="eyebrow"><Sparkles size={15} /> LOCAL-FIRST RESEARCH OS</div>
        <h1>把科研流程交给系统，<br /><em>把方向留给你。</em></h1>
        <p>检索前沿、验证研究空白、发现数据、生成实验与论文草稿。所有结论都有证据，所有实验都能复现。</p>
        <div className="promise-grid">
          <div><ShieldCheck /><span><b>本地优先</b>数据与密钥留在你的电脑</span></div>
          <div><Search /><span><b>四源检索</b>聚合最新论文并自动去重</span></div>
          <div><Code2 /><span><b>安全实验</b>隔离执行并导出复现包</span></div>
        </div>
      </section>
      <section className="auth-card-wrap">
        <form className="auth-card" onSubmit={submit}>
          <div className="auth-icon"><LockKeyhole size={24} /></div>
          <h2>{mode === "register" ? "创建本地研究账户" : "欢迎回来"}</h2>
          <p>单用户模式，不发送验证邮件，不依赖云端账户。</p>
          <label>邮箱<input value={email} onChange={(e) => setEmail(e.target.value)} type="email" required /></label>
          <label>密码<input value={password} onChange={(e) => setPassword(e.target.value)} type="password" minLength={8} required /></label>
          {error && <div className="error-note">{error}</div>}
          <button className="primary wide" disabled={busy}>
            {busy ? <LoaderCircle className="spin" /> : mode === "register" ? "进入工作台" : "登录"}
            {!busy && <ArrowRight size={18} />}
          </button>
          <button type="button" className="text-button" onClick={() => setMode(mode === "register" ? "login" : "register")}>
            {mode === "register" ? "已有账户？直接登录" : "第一次使用？创建账户"}
          </button>
        </form>
      </section>
    </main>
  );
}

function ModelModal({
  token,
  onClose,
  onSaved,
}: {
  token: string;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [form, setForm] = useState({
    name: "DeepSeek Research",
    provider: "deepseek",
    model: "deepseek-chat",
    api_key: "",
    base_url: "",
    budget_limit_usd: 5,
    input_price_per_million_usd: "",
    output_price_per_million_usd: "",
    is_default: true,
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function save(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api("/models", {
        method: "POST",
        body: JSON.stringify({
          ...form,
          input_price_per_million_usd:
            form.input_price_per_million_usd === ""
              ? null
              : Number(form.input_price_per_million_usd),
          output_price_per_million_usd:
            form.output_price_per_million_usd === ""
              ? null
              : Number(form.output_price_per_million_usd),
        }),
      }, token);
      onSaved();
      onClose();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "保存失败");
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="modal-backdrop">
      <form className="modal" onSubmit={save}>
        <button className="close" type="button" onClick={onClose}><X /></button>
        <div className="modal-kicker"><KeyRound size={16} /> BYOK MODEL</div>
        <h2>连接你的模型</h2>
        <p>密钥使用 AES-256-GCM 加密，仅保存在本地数据库。</p>
        <div className="form-grid">
          <label>配置名称<input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required /></label>
          <label>厂商
            <select value={form.provider} onChange={(e) => {
              const provider = e.target.value;
              setForm({
                ...form,
                provider,
                model: providerDefaultModels[provider] || "",
              });
            }}>
              {["deepseek", "mimo", "openai", "anthropic", "gemini", "openrouter", "qwen", "kimi", "glm", "minimax", "openai_compatible"].map((name) => <option key={name}>{name}</option>)}
            </select>
          </label>
          <label>模型 ID<input value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })} required /></label>
          <label>单任务预算上限（美元）<input type="number" min="0" step="0.5" value={form.budget_limit_usd} onChange={(e) => setForm({ ...form, budget_limit_usd: Number(e.target.value) })} /></label>
          <label>输入单价（美元/百万 Token）<input type="number" min="0" step="0.001" value={form.input_price_per_million_usd} onChange={(e) => setForm({ ...form, input_price_per_million_usd: e.target.value })} placeholder="留空使用内置价格" /></label>
          <label>输出单价（美元/百万 Token）<input type="number" min="0" step="0.001" value={form.output_price_per_million_usd} onChange={(e) => setForm({ ...form, output_price_per_million_usd: e.target.value })} placeholder="留空使用内置价格" /></label>
          <label className="span-2">API Key<input type="password" value={form.api_key} onChange={(e) => setForm({ ...form, api_key: e.target.value })} required placeholder="sk-..." /></label>
          {form.provider === "openai_compatible" && <label className="span-2">Base URL<input value={form.base_url} onChange={(e) => setForm({ ...form, base_url: e.target.value })} required /></label>}
        </div>
        {error && <div className="error-note">{error}</div>}
        <button className="primary wide" disabled={busy}>{busy ? <LoaderCircle className="spin" /> : <Check />}加密保存</button>
      </form>
    </div>
  );
}

function NewProjectModal({
  token,
  onClose,
  onCreated,
}: {
  token: string;
  onClose: () => void;
  onCreated: (project: Project) => void;
}) {
  const [title, setTitle] = useState("LLM 智能体评测研究");
  const [direction, setDirection] = useState("LLM agent evaluation");
  const [busy, setBusy] = useState(false);
  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    const project = await api<Project>("/projects", {
      method: "POST",
      body: JSON.stringify({ title, direction }),
    }, token);
    onCreated(project);
    onClose();
  }
  return (
    <div className="modal-backdrop">
      <form className="modal project-modal" onSubmit={submit}>
        <button className="close" type="button" onClick={onClose}><X /></button>
        <div className="modal-kicker"><Sparkles size={16} /> NEW RESEARCH</div>
        <h2>开启一个研究方向</h2>
        <p>系统会先检索前沿，再给出 3–5 个带证据的低覆盖候选课题。</p>
        <label>项目名称<input value={title} onChange={(e) => setTitle(e.target.value)} required /></label>
        <label>研究方向或关键词<textarea value={direction} onChange={(e) => setDirection(e.target.value)} rows={4} required /></label>
        <div className="scope-note"><ShieldCheck size={18} /><span>“研究空白”表示截至检索日期的低覆盖候选，不声称全球无人发表。</span></div>
        <button className="primary wide" disabled={busy}>{busy ? <LoaderCircle className="spin" /> : <Search />}创建并开始</button>
      </form>
    </div>
  );
}

function Metric({ label, value, suffix }: { label: string; value: string | number; suffix?: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}<small>{suffix}</small></strong></div>;
}

const validationLabel: Record<string, string> = {
  low_coverage_supported: "反向检索未发现新增记录",
  low_coverage_with_counterevidence: "发现少量潜在反证",
  contested: "候选存在较强竞争证据",
  inconclusive: "检索源不足，结论待确认",
};

function GapCard({
  gap,
  validation,
  selected,
  onSelect,
  onAdoptAlternative,
  busy,
}: {
  gap: Gap;
  validation?: GapValidation;
  selected: boolean;
  onSelect: () => void;
  onAdoptAlternative: (index: number) => void;
  busy: boolean;
}) {
  return (
    <article className={`gap-card ${selected ? "selected" : ""}`}>
      <div className="gap-top">
        <span className="confidence"><CircleDot size={14} /> 置信度 {Math.round(gap.confidence * 100)}%</span>
        {selected && <span className="selected-pill"><Check size={13} /> 已选定</span>}
      </div>
      {validation && (
        <div className={`validation-strip ${validation.status}`}>
          <ShieldCheck size={14} />
          <span>{validationLabel[validation.status] || validation.status}</span>
          <b>{validation.new_result_count} 篇新增命中</b>
        </div>
      )}
      <h3>{gap.title}</h3>
      <p className="hypothesis">{gap.hypothesis}</p>
      <p>{gap.rationale}</p>
      <div className="score-row">
        <span>新颖度 <b>{Math.round(gap.novelty_score * 100)}</b></span>
        <span>可行性 <b>{Math.round(gap.feasibility_score * 100)}</b></span>
        <span>成本 <b>{gap.estimated_cost.split("：")[0]}</b></span>
      </div>
      {gap.submission_readiness?.level && (
        <div className={`constraint-note ${gap.submission_readiness.passed ? "readiness-pass" : ""}`}>
          <b>
            {gap.submission_readiness.passed
              ? "投稿规划：具备继续实验的基础"
              : "投稿规划：当前选题暂不可直接投稿"}
          </b>
          {gap.submission_readiness.findings?.map((finding) => (
            <div key={finding}>• {finding}</div>
          ))}
          {!gap.submission_readiness.passed && gap.alternative_topics?.length > 0 && (
            <details>
              <summary>查看相似可行选题</summary>
              {gap.alternative_topics.map((topic, index) => (
                <div className="alternative-topic" key={topic.title}>
                  <strong>{topic.title}</strong>
                  <p>{topic.why_feasible}</p>
                  {topic.suggested_track && <small>建议目标：{topic.suggested_track}</small>}
                  <small>最低实验要求：{topic.minimum_experiment}</small>
                  <button
                    className="secondary wide"
                    disabled={busy}
                    onClick={() => onAdoptAlternative(index)}
                  >
                    采用这个相似选题
                  </button>
                </div>
              ))}
            </details>
          )}
          {gap.submission_readiness.details?.recommended_targets?.map((target) => (
            <details key={target.track}>
              <summary>目标建议：{target.track}（{target.fit}）</summary>
              <ul>{target.requirements.map((item) => <li key={item}>{item}</li>)}</ul>
              <small>{target.warning}</small>
            </details>
          ))}
        </div>
      )}
      <details><summary>风险与反向验证</summary>
        <ul>{gap.risks.map((risk) => <li key={risk}>{risk}</li>)}</ul>
        <div className="query-list">{gap.counter_queries.map((query) => <code key={query}>{query}</code>)}</div>
        {validation?.reverse_query_results.length ? (
          <div className="reverse-results">
            {validation.reverse_query_results.slice(0, 4).map((result) => (
              <a href={result.url} target="_blank" key={`${result.query}-${result.title}`}>
                <span>{result.source} · {result.publication_date || "日期未知"}</span>
                {result.title}
              </a>
            ))}
          </div>
        ) : validation && <p className="validation-note">截至 {new Date(validation.validated_at).toLocaleDateString("zh-CN")}，限定检索范围内未发现新的直接命中；这不等于全球无人发表。</p>}
      </details>
      {!selected && <button className="secondary wide" onClick={onSelect} disabled={busy}>选择这个课题 <ChevronRight size={17} /></button>}
    </article>
  );
}

function LiteratureView({
  papers,
  projectId,
  token,
}: {
  papers: Paper[];
  projectId: string;
  token: string;
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Paper[]>();
  const [searching, setSearching] = useState(false);
  async function semanticSearch(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) {
      setResults(undefined);
      return;
    }
    setSearching(true);
    try {
      setResults(await api<Paper[]>(
        `/projects/${projectId}/papers/semantic-search?q=${encodeURIComponent(query)}`,
        {},
        token,
      ));
    } finally {
      setSearching(false);
    }
  }
  const visiblePapers = results || papers;
  return (
    <div className="content-stack">
      <div className="section-head"><div><span className="eyebrow">LITERATURE MAP</span><h2>最新相关论文</h2></div><span className="count-pill">{papers.length} 篇去重记录</span></div>
      <form className="semantic-search" onSubmit={semanticSearch}>
        <Search size={18} />
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="用自然语言搜索：多语言智能体在工具故障下的恢复能力" />
        <button className="primary" disabled={searching}>{searching ? <LoaderCircle className="spin" /> : "语义检索"}</button>
        {results && <button type="button" className="ghost" onClick={() => { setQuery(""); setResults(undefined); }}>清除</button>}
      </form>
      <div className="paper-list">
        {visiblePapers.map((paper) => (
          <article className="paper-row" key={paper.id}>
            <div className="source-mark">{paper.source.slice(0, 2).toUpperCase()}</div>
            <div className="paper-copy">
              <div className="paper-meta"><span>{paper.source}</span><span>{paper.publication_date || "日期未知"}</span><span>引用 {paper.citation_count}</span>{paper.semantic_score !== undefined && <span>语义相关 {Math.round(paper.semantic_score * 100)}%</span>}{paper.open_access_url && <span className="oa">OPEN</span>}</div>
              <h3>{paper.title}</h3>
              <p>{paper.abstract || "该数据源未提供摘要。"}</p>
            </div>
            {(paper.open_access_url || paper.url) && <a className="icon-link" href={paper.open_access_url || paper.url} target="_blank"><ExternalLink size={18} /></a>}
          </article>
        ))}
        {!visiblePapers.length && <Empty icon={BookOpen} text="没有找到语义相关论文，可换一种描述。" />}
      </div>
    </div>
  );
}

function Empty({ icon: Icon, text }: { icon: typeof BookOpen; text: string }) {
  return <div className="empty"><Icon size={30} /><p>{text}</p></div>;
}

function Dashboard({ token, onLogout }: { token: string; onLogout: () => void }) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [readiness, setReadiness] = useState<Readiness>();
  const [selectedId, setSelectedId] = useState<string>();
  const [detail, setDetail] = useState<ProjectDetail>();
  const [view, setView] = useState<View>("overview");
  const [showModel, setShowModel] = useState(false);
  const [showProject, setShowProject] = useState(false);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");

  const loadBase = useCallback(async () => {
    try {
      const [projectRows, modelRows, readinessResult] = await Promise.all([
        api<Project[]>("/projects", {}, token),
        api<ModelConfig[]>("/models", {}, token),
        api<Readiness>("/readiness", {}, token),
      ]);
      setProjects(projectRows);
      setModels(modelRows);
      setReadiness(readinessResult);
      if (!selectedId && projectRows[0]) setSelectedId(projectRows[0].id);
    } catch (reason) {
      if (reason instanceof ApiError && reason.status === 401) onLogout();
    }
  }, [token, onLogout, selectedId]);

  const loadDetail = useCallback(async () => {
    if (!selectedId) return;
    try {
      const data = await api<ProjectDetail>(`/projects/${selectedId}`, {}, token);
      setDetail(data);
      setProjects((rows) => rows.map((row) => row.id === data.project.id ? data.project : row));
    } catch (reason) {
      if (reason instanceof ApiError && reason.status === 401) onLogout();
    }
  }, [selectedId, token, onLogout]);

  useEffect(() => { void loadBase(); }, [loadBase]);
  useEffect(() => {
    void loadDetail();
    if (!selectedId) return;
    const unsubscribe = subscribeProjectEvents(
      selectedId,
      token,
      () => void loadDetail(),
    );
    const fallback = window.setInterval(() => void loadDetail(), 15000);
    return () => {
      unsubscribe();
      window.clearInterval(fallback);
    };
  }, [loadDetail, selectedId, token]);

  const selected = detail?.project;
  const selectedGap = detail?.gaps.find((gap) => gap.id === selected?.selected_gap_id);
  const activeModel = models.find((item) => item.is_default) || models[0];
  const modelSpend = (detail?.model_calls || []).reduce((sum, call) => sum + (call.cost_usd || 0), 0);
  const progress = useMemo(() => {
    const status = selected?.status;
    return status === "draft" ? 8 : status === "discovering" ? 35 : status === "awaiting_topic" ? 55 : status === "planning" ? 75 : status === "paused" ? 50 : status === "ready" ? 100 : 0;
  }, [selected?.status]);

  async function action(name: string, fn: () => Promise<unknown>) {
    setBusy(name);
    setNotice("");
    try {
      await fn();
      setNotice("任务已提交，工作台会自动更新。");
      await loadDetail();
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : "操作失败");
    } finally {
      setBusy("");
    }
  }

  async function discover() {
    if (!selected) return;
    await action("discover", () => api(`/projects/${selected.id}/discover`, { method: "POST" }, token));
  }

  async function togglePause() {
    if (!selected) return;
    const paused = selected.status === "paused";
    await action(paused ? "resume" : "pause", () => api(
      `/projects/${selected.id}/${paused ? "resume" : "pause"}`,
      { method: "POST" },
      token,
    ));
  }

  async function chooseGap(gap: Gap) {
    if (!selected) return;
    await action(`gap-${gap.id}`, () => api(`/projects/${selected.id}/select-gap`, {
      method: "POST",
      body: JSON.stringify({ gap_id: gap.id }),
    }, token));
  }

  async function adoptAlternativeTopic(gap: Gap, alternativeIndex: number) {
    if (!selected) return;
    await action(`alternative-${gap.id}-${alternativeIndex}`, () => api(
      `/projects/${selected.id}/adopt-alternative-topic`,
      {
        method: "POST",
        body: JSON.stringify({
          source_gap_id: gap.id,
          alternative_index: alternativeIndex,
        }),
      },
      token,
    ));
  }

  async function generateManuscript(
    target: string,
    mode: "draft" | "submission",
    publicationName?: string,
    authorGuideUrl?: string,
    venueEvidenceUrl?: string,
    venueClaim?: string,
    venueVerifiedOn?: string,
    venueHumanVerified?: boolean,
  ) {
    if (!selected) return;
    await action("manuscript", () => api(`/projects/${selected.id}/manuscript`, {
      method: "POST",
      body: JSON.stringify({
        target,
        mode,
        publication_name: publicationName || null,
        author_guide_url: authorGuideUrl || null,
        venue_evidence_url: venueEvidenceUrl || null,
        venue_claim: venueClaim || null,
        venue_verified_on: venueVerifiedOn || null,
        venue_human_verified: Boolean(venueHumanVerified),
      }),
    }, token));
  }

  async function testDefaultModel() {
    const model = models.find((item) => item.is_default) || models[0];
    if (!model) {
      setShowModel(true);
      return;
    }
    await action("model-test", () => api(`/models/${model.id}/test`, {
      method: "POST",
    }, token));
    await loadBase();
  }

  async function runExperiment() {
    if (!selected) return;
    await action("experiment", () => api(`/projects/${selected.id}/run-experiment`, {
      method: "POST",
      body: JSON.stringify({ allow_network: false, timeout_seconds: 300 }),
    }, token));
  }

  async function confirmDatasetValidity(dataset: Dataset) {
    if (!selected) return;
    const reason = window.prompt(
      "该数据集与课题的自动适配审查未通过。请说明仍要继续使用它的科学理由（至少 10 个字符）：",
    );
    if (!reason || reason.trim().length < 10) return;
    await action("dataset-confirm", () => api(
      `/projects/${selected.id}/confirm-dataset-validity`,
      {
        method: "POST",
        body: JSON.stringify({
          dataset_id: dataset.id,
          confirmed: true,
          reason: reason.trim(),
        }),
      },
      token,
    ));
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Logo />
        <nav>
          <span className="nav-label">研究空间</span>
          {navItems.map((item) => {
            const Icon = item.icon;
            return <button key={item.id} className={view === item.id ? "active" : ""} onClick={() => setView(item.id)}><Icon size={18} />{item.label}</button>;
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="local-card"><ShieldCheck size={18} /><div><b>本地模式</b><span>零云基础设施费用</span></div><span className="online-dot" /></div>
          <button className="logout" onClick={onLogout}><LogOut size={17} />退出本地账户</button>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div className="project-switcher">
            <span>当前项目</span>
            <select value={selectedId || ""} onChange={(e) => setSelectedId(e.target.value)}>
              {!projects.length && <option value="">尚无项目</option>}
              {projects.map((project) => <option value={project.id} key={project.id}>{project.title}</option>)}
            </select>
          </div>
          <div className="top-actions">
            <button
              className="ghost"
              title={readiness?.next_action || "本地基础设施与模型连接均已通过"}
              onClick={() => {
                if (!readiness?.checks.model_reachable) setShowModel(true);
              }}
            >
              <ShieldCheck size={17} />
              {readiness?.acceptance_ready
                ? "验收就绪"
                : readiness?.software_ready
                  ? "待连接模型"
                  : "检查本地环境"}
            </button>
            {models.length > 0 && (
              <button
                className="ghost"
                onClick={testDefaultModel}
                disabled={busy === "model-test"}
                title={activeModel
                  ? `累计 $${activeModel.spent_usd.toFixed(6)} / 上限 $${activeModel.budget_limit_usd.toFixed(2)}`
                  : undefined}
              >
                {busy === "model-test" ? <LoaderCircle className="spin" size={17} /> : <CircleDot size={17} />}
                {models.some((model) => model.capabilities?.reachable)
                  ? "模型已连通"
                  : "测试模型"}
              </button>
            )}
            <button className="ghost" onClick={() => setShowModel(true)}><KeyRound size={17} />{models.length ? `${models.length} 个模型` : "配置模型"}</button>
            <button className="primary" onClick={() => setShowProject(true)}><Plus size={18} />新研究</button>
          </div>
        </header>

        {!selected ? (
          <section className="welcome-empty">
            <div className="orb"><Sparkles size={36} /></div>
            <span className="eyebrow">YOUR FIRST RESEARCH RUN</span>
            <h1>从一个方向开始</h1>
            <p>输入你关注的研究方向，ResearchFlow 会建立论文证据图谱，并给出可验证的低覆盖课题。</p>
            <button className="primary" onClick={() => setShowProject(true)}><Plus />创建第一个研究项目</button>
          </section>
        ) : (
          <section className="canvas">
            <div className="hero">
              <div>
                <span className="eyebrow">RESEARCH PROJECT · {statusLabel[selected.status] || selected.status}</span>
                <h1>{selected.title}</h1>
                <p>{selected.direction}</p>
              </div>
              <div className="hero-actions">
                {["discovering", "planning", "paused"].includes(selected.status) && (
                  <button className="ghost" onClick={togglePause} disabled={busy === "pause" || busy === "resume"}>
                    {selected.status === "paused" ? "恢复流程" : "暂停流程"}
                  </button>
                )}
                <button className="secondary" onClick={discover} disabled={busy === "discover" || selected.status === "paused"}>
                  {busy === "discover" ? <LoaderCircle className="spin" /> : <RefreshCw />} {detail?.papers.length ? "刷新前沿" : "开始检索"}
                </button>
              </div>
            </div>

            <div className="progress-panel">
              <div className="progress-copy"><span>自动科研流程</span><b>{progress}%</b></div>
              <div className="progress-track"><i style={{ width: `${progress}%` }} /></div>
              <div className="progress-steps">
                {["聚合论文", "提取证据", "验证空白", "匹配数据", "生成实验"].map((step, index) => <span className={progress >= (index + 1) * 20 ? "done" : ""} key={step}><i>{progress >= (index + 1) * 20 ? <Check size={11} /> : index + 1}</i>{step}</span>)}
              </div>
            </div>
            {notice && <div className="notice">{notice}</div>}

            {view === "overview" && (
              <div className="overview-grid">
                <section className="main-column">
                  <div className="metrics-row">
                    <Metric label="去重论文" value={detail?.papers.length || 0} suffix="篇" />
                    <Metric label="证据片段" value={detail?.evidence.length || 0} suffix="条" />
                    <Metric label="候选课题" value={detail?.gaps.length || 0} suffix="个" />
                    <Metric label="公开数据" value={detail?.datasets.length || 0} suffix="组" />
                  </div>
                  <div className="panel">
                    <div className="panel-head"><div><span className="eyebrow">NEXT DECISION</span><h2>{selectedGap ? "已选定研究课题" : "等待你的方向选择"}</h2></div><Target /></div>
                    {selectedGap ? (
                      <div className="selected-topic"><span>LOW-COVERAGE CANDIDATE</span><h3>{selectedGap.title}</h3><p>{selectedGap.hypothesis}</p><button className="text-link" onClick={() => setView("gaps")}>查看证据与风险 <ArrowRight size={15} /></button></div>
                    ) : detail?.gaps.length ? (
                      <><p>系统已根据检索快照生成 {detail.gaps.length} 个候选。选择后将自动寻找数据并生成实验包。</p><button className="primary" onClick={() => setView("gaps")}>审阅候选课题 <ArrowRight /></button></>
                    ) : (
                      <><p>先运行前沿检索。无需模型密钥也能完成证据聚合与基础候选生成。</p><button className="primary" onClick={discover}><Search />开始前沿检索</button></>
                    )}
                  </div>
                  <div className="panel">
                    <div className="panel-head"><div><span className="eyebrow">LATEST PAPERS</span><h2>近期论文快照</h2></div><button className="text-link" onClick={() => setView("literature")}>全部论文 <ArrowRight size={15} /></button></div>
                    <div className="mini-papers">{detail?.papers.slice(0, 4).map((paper) => <div key={paper.id}><span>{paper.publication_date || paper.source}</span><b>{paper.title}</b><em>{paper.source}</em></div>)}{!detail?.papers.length && <Empty icon={BookOpen} text="检索完成后，这里会展示最新论文。" />}</div>
                  </div>
                </section>
                <aside className="activity-column">
                  <div className="panel activity-panel">
                    <div className="panel-head"><div><span className="eyebrow">PROVENANCE</span><h2>证据链活动</h2></div><ShieldCheck /></div>
                    <div className="timeline">{detail?.events.map((event) => <div key={event.id}><i /><span>{new Date(event.created_at).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}</span><p><b>{event.stage}</b>{event.message}</p></div>)}</div>
                  </div>
                  <div className="panel security-panel"><LockKeyhole /><div><b>研究边界</b><p>候选课题只表示当前检索范围内覆盖较低。系统保留反向检索式，不伪造文献与实验结果。</p></div></div>
                  <div className="panel model-ledger">
                    <div className="panel-head"><div><span className="eyebrow">MODEL LEDGER</span><h2>模型调用账本</h2></div><KeyRound /></div>
                    <strong>${modelSpend.toFixed(6)}</strong>
                    <span>当前项目已记录费用</span>
                    {activeModel && (
                      <p>
                        默认模型累计 ${activeModel.spent_usd.toFixed(6)}，
                        剩余额度 ${activeModel.remaining_budget_usd.toFixed(6)}
                        （上限 ${activeModel.budget_limit_usd.toFixed(2)}）
                      </p>
                    )}
                    {(detail?.model_calls || []).slice(0, 5).map((call) => (
                      <div className="model-call" key={call.id}>
                        <b>{call.purpose}</b>
                        <span>{call.provider} / {call.model}</span>
                        <code>{(call.input_tokens || 0) + (call.output_tokens || 0)} tokens · ${(call.cost_usd || 0).toFixed(6)}</code>
                      </div>
                    ))}
                    {!detail?.model_calls?.length && <p>尚未使用已配置的大模型，当前流程使用可审计离线回退。</p>}
                  </div>
                </aside>
              </div>
            )}

            {view === "literature" && <LiteratureView papers={detail?.papers || []} projectId={selected.id} token={token} />}
            {view === "gaps" && <div className="content-stack">
              <div className="section-head"><div><span className="eyebrow">GAP VALIDATION</span><h2>低覆盖研究空白候选</h2><p>每个候选都已实际执行反向检索，并展示潜在反证与检索日期。</p></div></div>
              {detail?.coverage_matrix && (
                <div className="coverage-panel">
                  {Object.entries(detail.coverage_matrix.summary).map(([dimension, values]) => (
                    <div key={dimension}>
                      <b>{({ tasks: "任务", methods: "方法", datasets: "数据", metrics: "指标" } as Record<string, string>)[dimension] || dimension}</b>
                      <div>{Object.entries(values).slice(0, 6).map(([name, count]) => <span key={name}>{name.replaceAll("_", " ")} <em>{count}</em></span>)}</div>
                    </div>
                  ))}
                </div>
              )}
              <div className="gap-grid">{detail?.gaps.map((gap) => <GapCard key={gap.id} gap={gap} validation={detail.gap_validations.find((item) => item.gap_id === gap.id)} selected={gap.id === selected.selected_gap_id} onSelect={() => chooseGap(gap)} onAdoptAlternative={(index) => adoptAlternativeTopic(gap, index)} busy={busy === `gap-${gap.id}` || busy.startsWith(`alternative-${gap.id}-`)} />)}{!detail?.gaps.length && <Empty icon={Target} text="先启动论文检索，系统将生成候选课题。" />}</div>
            </div>}
            {view === "data" && (
              <DatasetView
                datasets={detail?.datasets || []}
                preparations={detail?.data_preparations || []}
                onConfirm={confirmDatasetValidity}
                busy={busy === "dataset-confirm"}
              />
            )}
            {view === "experiment" && (
              <ExperimentView
                experiment={detail?.experiments.at(-1)}
                preparation={detail?.data_preparations.at(-1)}
                runs={detail?.experiment_runs || []}
                busy={busy === "experiment"}
                onRun={runExperiment}
                onDownload={() => downloadArtifact(selected.id, "experiment", token)}
              />
            )}
            {view === "paper" && (
              <ManuscriptView
                builds={detail?.manuscripts || []}
                hasCompletedRun={Boolean(detail?.experiment_runs.some(
                  (run) => run.status === "completed" && run.quality_level === "reproducible_research",
                ))}
                busy={busy === "manuscript"}
                onGenerate={generateManuscript}
                onDownload={() => downloadArtifact(selected.id, "manuscript", token)}
              />
            )}
          </section>
        )}
      </main>
      {showModel && <ModelModal token={token} onClose={() => setShowModel(false)} onSaved={loadBase} />}
      {showProject && <NewProjectModal token={token} onClose={() => setShowProject(false)} onCreated={(project) => { setProjects((rows) => [project, ...rows]); setSelectedId(project.id); setView("overview"); setTimeout(() => { void api(`/projects/${project.id}/discover`, { method: "POST" }, token).catch(() => undefined); }, 200); }} />}
    </div>
  );
}

function DatasetView({
  datasets,
  preparations,
  onConfirm,
  busy,
}: {
  datasets: Dataset[];
  preparations: DataPreparation[];
  onConfirm: (dataset: Dataset) => void;
  busy: boolean;
}) {
  const preparationByDataset = new Map(
    preparations.map((item) => [item.dataset_id, item]),
  );
  return (
    <div className="content-stack">
      <div className="section-head">
        <div>
          <span className="eyebrow">LICENSED DATA</span>
          <h2>公开数据集与处理记录</h2>
          <p>只自动处理许可白名单中的公开数据，并保留样本、变换与内容指纹。</p>
        </div>
      </div>
      <div className="dataset-grid">
        {datasets.map((dataset) => {
          const preparation = preparationByDataset.get(dataset.id);
          return (
            <article className={`dataset-card ${preparation ? "selected-data" : ""}`} key={dataset.id}>
              <div className="dataset-icon"><Database /></div>
              <span>{dataset.source}{preparation ? " · AUTO SELECTED" : ""}</span>
              <h3>{dataset.name}</h3>
              <p>{dataset.quality_notes}</p>
              <div className="license"><ShieldCheck size={15} />许可：{dataset.license || "待人工确认"}</div>
              {dataset.validity_audit?.level && (
                <div className="constraint-note">
                  科学适配：{QUALITY_LABELS[dataset.validity_audit.level] || dataset.validity_audit.level}
                  {dataset.validity_audit.findings?.map((finding) => (
                    <div key={finding}>• {finding}</div>
                  ))}
                  {dataset.validity_audit.details?.baseline_paths?.map((path) => (
                    <div key={path.path}>
                      {path.passed ? "✓" : "×"} {path.label}：{path.required}
                    </div>
                  ))}
                </div>
              )}
              {dataset.validity_audit?.passed === false && !dataset.human_confirmed && (
                <button
                  className="secondary"
                  disabled={busy}
                  onClick={() => onConfirm(dataset)}
                >
                  人工确认风险并继续
                </button>
              )}
              {preparation && (
                <div className="data-proof">
                  <b>{preparation.row_count} 条可复现样本</b>
                  <span>{preparation.config_name} / {preparation.split_name}</span>
                  <code title={preparation.content_hash}>
                    SHA-256 {preparation.content_hash?.slice(0, 16)}…
                  </code>
                </div>
              )}
              <a href={dataset.url} target="_blank">查看数据集 <ExternalLink size={15} /></a>
            </article>
          );
        })}
        {!datasets.length && <Empty icon={Database} text="选择研究课题后，系统会自动匹配并处理数据集。" />}
      </div>
    </div>
  );
}

function ExperimentView({
  experiment,
  preparation,
  runs,
  busy,
  onRun,
  onDownload,
}: {
  experiment?: Experiment;
  preparation?: DataPreparation;
  runs: ExperimentRun[];
  busy: boolean;
  onRun: () => void;
  onDownload: () => void;
}) {
  const latestRun = runs[0];
  const codeOrigin = String(experiment?.resource_profile.code_origin || "unknown");
  const baselinePaths = Array.isArray(experiment?.resource_profile.baseline_paths)
    ? experiment?.resource_profile.baseline_paths as Array<{
      path: string;
      label: string;
      passed: boolean;
      required: string;
    }>
    : [];
  const terminalOutput = experiment
    ? [
        "$ docker run --network none --cpus 2 --memory 4g",
        `✓ dataset rows: ${preparation?.row_count || 0}`,
        `✓ dataset sha256: ${preparation?.content_hash?.slice(0, 20) || "pending"}…`,
        `✓ code origin: ${codeOrigin}`,
        `✓ fixed seed: 42`,
        latestRun
          ? `→ latest run: ${latestRun.status}\n${JSON.stringify(latestRun.results, null, 2)}`
          : "→ ready for isolated execution",
      ].join("\n")
    : "$ waiting for a selected research gap...";
  return (
    <div className="content-stack">
      <div className="section-head">
        <div>
          <span className="eyebrow">REPRODUCIBLE RUN</span>
          <h2>安全实验工作台</h2>
          <p>非 root、默认断网、限制 CPU / 内存 / 进程数与执行时间，超时后强制清理容器。</p>
        </div>
      </div>
      <div className="lab-panel">
        <div className="terminal">
          <div className="terminal-bar"><i /><i /><i /><span>researchflow / sandbox</span></div>
          <pre>{terminalOutput}</pre>
        </div>
        <div className="lab-actions">
          <div>
            <Box />
            <h3>{experiment ? experiment.name : "尚未生成实验"}</h3>
            <p>{experiment?.objective || "选择课题后系统会处理数据、生成安全代码与复现实验包。"}</p>
            {experiment && (
              <div className="run-facts">
                <span>证据等级 <b>{QUALITY_LABELS[latestRun?.quality_level || experiment.quality_level] || latestRun?.quality_level || experiment.quality_level}</b></span>
                <span>代码来源 <b>{codeOrigin === "llm" ? "大模型生成并通过 AST 审查" : "可审计离线回退"}</b></span>
                <span>网络权限 <b>关闭</b></span>
                <span>最近运行 <b>{latestRun?.status || "尚未执行"}</b></span>
              </div>
            )}
            {baselinePaths.length > 0 && (
              <div className="constraint-note">
                <b>可投稿实验路径</b>
                {baselinePaths.map((path) => (
                  <div key={path.path}>
                    {path.passed ? "✓" : "×"} {path.label}：{path.required}
                  </div>
                ))}
              </div>
            )}
          </div>
          <button className="primary" disabled={!experiment || busy} onClick={onRun}>
            {busy ? <LoaderCircle className="spin" /> : <Zap />}安全运行
          </button>
          <button className="secondary" disabled={!experiment} onClick={onDownload}>
            <Download />下载实验包
          </button>
        </div>
      </div>
    </div>
  );
}

function ManuscriptView({
  builds,
  hasCompletedRun,
  busy,
  onGenerate,
  onDownload,
}: {
  builds: Array<{
    id: string;
    target: string;
    status: string;
    mode: string;
    quality_level: string;
    validity_audit: {
      passed?: boolean;
      findings?: string[];
      pre_submission_review?: {
        passed: boolean;
        recommendation: string;
        summary: { critical: number; major: number; minor: number };
        findings: Array<{
          severity: string;
          category: string;
          message: string;
          action: string;
        }>;
      };
    };
  }>;
  hasCompletedRun: boolean;
  busy: boolean;
  onGenerate: (
    target: string,
    mode: "draft" | "submission",
    publicationName?: string,
    authorGuideUrl?: string,
    venueEvidenceUrl?: string,
    venueClaim?: string,
    venueVerifiedOn?: string,
    venueHumanVerified?: boolean,
  ) => void;
  onDownload: () => void;
}) {
  const [target, setTarget] = useState("arxiv");
  const [mode, setMode] = useState<"draft" | "submission">("draft");
  const [publicationName, setPublicationName] = useState("");
  const [authorGuideUrl, setAuthorGuideUrl] = useState("");
  const [venueEvidenceUrl, setVenueEvidenceUrl] = useState("");
  const [venueClaim, setVenueClaim] = useState("");
  const [venueVerifiedOn, setVenueVerifiedOn] = useState("");
  const [venueHumanVerified, setVenueHumanVerified] = useState(false);
  const requiresPublication = ["ieee_conference", "elsevier_journal"].includes(target);
  const latest = builds.at(-1);
  return (
    <div className="content-stack">
      <div className="section-head">
        <div>
          <span className="eyebrow">MANUSCRIPT BUILD</span>
          <h2>英文论文工程</h2>
          <p>LaTeX、BibTeX 与声明级溯源；投稿模式必须绑定真实完成的实验结果。</p>
        </div>
      </div>
      <div className="manuscript-panel">
        <div className="paper-preview">
          <div className="paper-sheet">
            <span>{target === "arxiv" ? "PREPRINT" : "ANONYMOUS SUBMISSION"}</span>
            <h2>Evidence-Grounded Research Draft</h2>
            <p className="authors">Anonymous Authors</p>
            <h3>Abstract</h3>
            <p>Every citation and numerical claim is linked to the project evidence store or a completed sandbox run.</p>
            <h3>1 &nbsp; Introduction</h3>
            <div className="fake-lines" /><div className="fake-lines short" />
          </div>
        </div>
        <div className="build-card">
          <FileText size={28} />
          <h3>{latest ? `${latest.target.toUpperCase()} 工程已生成` : "生成论文工程"}</h3>
          {latest && (
            <div className="constraint-note">
              证据等级：{QUALITY_LABELS[latest.quality_level] || latest.quality_level}
              {latest.validity_audit?.pre_submission_review && (
                <>
                  <div>
                    投稿前预审：
                    {latest.validity_audit.pre_submission_review.passed
                      ? "通过"
                      : latest.validity_audit.pre_submission_review.recommendation === "major_revision"
                        ? "需要重大修改"
                        : "已阻止投稿候选"}
                  </div>
                  {latest.validity_audit.pre_submission_review.findings.map((finding, index) => (
                    <div key={`${finding.category}-${index}`}>
                      [{finding.severity}] {finding.message} 修改建议：{finding.action}
                    </div>
                  ))}
                </>
              )}
              {latest.validity_audit?.findings?.map((finding) => (
                <div key={finding}>• {finding}</div>
              ))}
            </div>
          )}
          <p>{latest ? (latest.status === "completed" ? "LaTeX、BibTeX 与 PDF 已完成。" : "LaTeX 与 BibTeX 已完成；本机缺少 LaTeX 编译器。") : "没有完成实验时只能生成明确标注的研究草稿。"}</p>
          <div className="build-controls">
            <label>目标模板
              <select value={target} onChange={(event) => setTarget(event.target.value)}>
                <option value="arxiv">arXiv</option>
                <option value="iclr">ICLR</option>
                <option value="icml">ICML</option>
                <option value="neurips">NeurIPS</option>
                <option value="ieee_conference">IEEE / EI 会议</option>
                <option value="elsevier_journal">Elsevier / SCI 期刊</option>
              </select>
            </label>
            <label>生成模式
              <select value={mode} onChange={(event) => setMode(event.target.value as "draft" | "submission")}>
                <option value="draft">研究草稿</option>
                <option value="submission" disabled={!hasCompletedRun}>结果投稿稿</option>
              </select>
            </label>
          </div>
          {requiresPublication && (
            <div className="build-controls publication-controls">
              <label>具体期刊或会议名称
                <input
                  value={publicationName}
                  onChange={(event) => setPublicationName(event.target.value)}
                  placeholder="例如：具体 IEEE 会议全称"
                />
              </label>
              <label>官方作者指南网址
                <input
                  value={authorGuideUrl}
                  onChange={(event) => setAuthorGuideUrl(event.target.value)}
                  placeholder="https://..."
                />
              </label>
              <label>索引或分区证据网址
                <input
                  value={venueEvidenceUrl}
                  onChange={(event) => setVenueEvidenceUrl(event.target.value)}
                  placeholder="https://..."
                />
              </label>
              <label>核验类别
                <select value={venueClaim} onChange={(event) => setVenueClaim(event.target.value)}>
                  <option value="">请选择</option>
                  <option value="sci_q3">SCI 三区</option>
                  <option value="sci_q4">SCI 四区</option>
                  <option value="ei_conference">EI 会议</option>
                  <option value="other">其他</option>
                </select>
              </label>
              <label>核验日期
                <input
                  type="date"
                  value={venueVerifiedOn}
                  onChange={(event) => setVenueVerifiedOn(event.target.value)}
                />
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={venueHumanVerified}
                  onChange={(event) => setVenueHumanVerified(event.target.checked)}
                />
                我已人工核对当前年份、数据库与学科分类
              </label>
            </div>
          )}
          {requiresPublication && mode === "submission" && (!publicationName || !authorGuideUrl) && (
            <div className="constraint-note">
              投稿模式必须指定具体出版物和官方作者指南；“SCI/EI”本身不是一种统一模板。
            </div>
          )}
          {!hasCompletedRun && <div className="constraint-note">尚无完成的实验运行，结果投稿模式已锁定。</div>}
          <ul>
            <li><Check />引用键与论文证据绑定</li>
            <li><Check />数字只来自已完成运行</li>
            <li><Check />附带 claim-provenance.json</li>
          </ul>
          <button
            className="primary wide"
            onClick={() => onGenerate(
              target,
              mode,
              publicationName,
              authorGuideUrl,
              venueEvidenceUrl,
              venueClaim,
              venueVerifiedOn,
              venueHumanVerified,
            )}
            disabled={busy || (
              requiresPublication
              && mode === "submission"
              && (
                !publicationName
                || !authorGuideUrl
                || !venueEvidenceUrl
                || !venueClaim
                || !venueVerifiedOn
                || !venueHumanVerified
              )
            )}
          >
            {busy ? <LoaderCircle className="spin" /> : <Sparkles />}
            生成 {target.toUpperCase()} {mode === "draft" ? "草稿" : "投稿稿"}
          </button>
          <button className="secondary wide" disabled={!latest} onClick={onDownload}>
            <Download />下载论文工程
          </button>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const [token, setToken] = useState<string>();
  useEffect(() => setToken(localStorage.getItem("researchflow_token") || undefined), []);
  if (!token) return <AuthScreen onAuthenticated={setToken} />;
  return <Dashboard token={token} onLogout={() => { localStorage.removeItem("researchflow_token"); setToken(undefined); }} />;
}
