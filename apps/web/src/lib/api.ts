const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function subscribeProjectEvents(
  projectId: string,
  token: string,
  onMessage: () => void,
) {
  const controller = new AbortController();
  void (async () => {
    const response = await fetch(`${API_URL}/projects/${projectId}/events`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok || !response.body) return;
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (!controller.signal.aborted) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";
      for (const frame of frames) {
        if (frame.split("\n").some((line) => line.startsWith("data:"))) {
          onMessage();
        }
      }
    }
  })().catch((reason: unknown) => {
    if (!controller.signal.aborted) {
      console.warn("ResearchFlow event stream disconnected", reason);
    }
  });
  return () => controller.abort();
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T>(
  path: string,
  options: RequestInit = {},
  token?: string,
): Promise<T> {
  const headers = new Headers(options.headers);
  if (!(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers,
    cache: "no-store",
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = data.detail;
    const message = typeof detail === "string"
      ? detail
      : detail?.message
        ? `${detail.message}${detail.findings?.length ? `：${detail.findings.join("；")}` : ""}`
        : "请求失败";
    throw new ApiError(response.status, message);
  }
  return response.json();
}

export async function downloadArtifact(
  projectId: string,
  kind: "experiment" | "manuscript",
  token: string,
) {
  const response = await fetch(
    `${API_URL}/projects/${projectId}/artifacts/${kind}`,
    { headers: { Authorization: `Bearer ${token}` } },
  );
  if (!response.ok) throw new ApiError(response.status, "产物尚未生成");
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${kind}.zip`;
  anchor.click();
  URL.revokeObjectURL(url);
}

export type Project = {
  id: string;
  title: string;
  direction: string;
  workflow_mode?: string;
  target_track?: string;
  topic_flexibility?: string;
  human_checkpoints?: string;
  status: string;
  selected_gap_id?: string;
  created_at: string;
  updated_at: string;
};

export type Gap = {
  id: string;
  title: string;
  hypothesis: string;
  rationale: string;
  confidence: number;
  novelty_score: number;
  feasibility_score: number;
  estimated_cost: string;
  risks: string[];
  counter_queries: string[];
  submission_readiness: {
    passed?: boolean;
    score?: number;
    level?: string;
    findings?: string[];
    details?: {
      stage?: string;
      satisfied_conditions?: string[];
      pending_conditions?: string[];
      blocker_type?: string | null;
      target_track?: string;
      estimated_resources?: {
        memory_gb?: number;
        gpu?: string;
        model_budget_usd?: number;
      };
      usable_dataset_count?: number;
      prepared_rows?: number;
      meaning?: string;
      recommended_targets?: Array<{
        track: string;
        fit: string;
        requirements: string[];
        warning: string;
      }>;
    };
  };
  alternative_topics: Array<{
    title: string;
    why_feasible: string;
    minimum_experiment: string;
    suggested_track?: string;
    addresses?: string[];
  }>;
};

export type GapValidation = {
  id: string;
  gap_id: string;
  status: string;
  initial_confidence: number;
  validated_confidence: number;
  reverse_query_results: Array<{
    query: string;
    title: string;
    source: string;
    publication_date?: string;
    url?: string;
  }>;
  new_result_count: number;
  counterevidence_count: number;
  search_errors: string[];
  validated_at: string;
};

export type Paper = {
  id: string;
  title: string;
  abstract: string;
  source: string;
  publication_date?: string;
  citation_count: number;
  open_access_url?: string;
  url?: string;
  semantic_score?: number;
};

export type Dataset = {
  id: string;
  name: string;
  source: string;
  url: string;
  license?: string;
  size_hint?: string;
  quality_notes: string;
  metadata_json: Record<string, unknown>;
  validity_audit: {
    passed?: boolean;
    level?: string;
    findings?: string[];
    details?: {
      baseline_paths?: BaselinePath[];
      [key: string]: unknown;
    };
  };
  human_confirmed: boolean;
};

export type BaselinePath = {
  path: string;
  label: string;
  passed: boolean;
  required: string;
  evidence: Record<string, unknown>;
};

export type DataPreparation = {
  id: string;
  dataset_id: string;
  status: string;
  config_name?: string;
  split_name?: string;
  row_count: number;
  schema_json: Record<string, unknown>;
  profile_json: Record<string, unknown>;
  transformations: string[];
  content_hash?: string;
  artifact_path?: string;
};

export type Experiment = {
  id: string;
  name: string;
  objective: string;
  artifact_path?: string;
  resource_profile: Record<string, unknown>;
  scientific_plan: Record<string, unknown>;
  validity_audit: {
    passed?: boolean;
    level?: string;
    findings?: string[];
  };
  quality_level: string;
};

export type ExperimentRun = {
  id: string;
  spec_id: string;
  status: string;
  started_at?: string;
  finished_at?: string;
  results: Record<string, unknown>;
  logs_path?: string;
  validity_audit: {
    passed?: boolean;
    level?: string;
    findings?: string[];
  };
  quality_level: string;
};

export type Event = {
  id: string;
  stage: string;
  message: string;
  created_at: string;
  payload: Record<string, unknown>;
};

export type ProjectDetail = {
  project: Project;
  papers: Paper[];
  evidence: Array<{
    id: string;
    claim: string;
    excerpt: string;
    locator?: string;
  }>;
  gaps: Gap[];
  gap_validations: GapValidation[];
  coverage_matrix?: {
    id: string;
    dimensions: Record<string, string[]>;
    rows: Array<Record<string, unknown>>;
    summary: Record<string, Record<string, number>>;
    created_at: string;
  };
  datasets: Dataset[];
  data_preparations: DataPreparation[];
  experiments: Experiment[];
  experiment_runs: ExperimentRun[];
  manuscripts: Array<{
    id: string;
    target: string;
    status: string;
    artifact_path?: string;
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
      manuscript_compilation?: {
        passed: boolean;
        status: string;
        message: string;
      };
    };
  }>;
  workflow_checkpoints: Array<{
    id: string;
    workflow_type: string;
    stage: string;
    status: string;
    requires_action: boolean;
    state: Record<string, unknown>;
    created_at: string;
  }>;
  model_calls: Array<{
    id: string;
    provider: string;
    model: string;
    purpose: string;
    status: string;
    input_tokens?: number;
    output_tokens?: number;
    cost_usd?: number;
    created_at: string;
  }>;
  events: Event[];
};
