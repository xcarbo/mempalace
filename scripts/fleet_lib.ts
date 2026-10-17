// fleet_lib.ts: the pure parts of scripts/fleet (config, diff classifier,
// key=value parsing, table rendering). No I/O here, so it is unit-tested in
// scripts/fleet.test.ts.

export type Kind = "local" | "box";

export interface BoxPaths {
  venv_link: string;
  src_link: string;
  api_link: string;
  python: string;
  expect_python: string;
  expect_sqlite: string;
  api_unit: string;
  api_port: number;
  keep: number;
  extra_packages: string[];
}

export interface LocalPaths {
  mempalace_bin: string;
  python: string;
  api_port: number;
  backup_dir: string;
  backup_glob: string;
  backup_log: string;
}

export interface Server {
  name: string;
  kind: Kind;
  enabled: boolean;
  note: string;
  ssh: string;
  backup_timer: string;
  box: BoxPaths;
  local: LocalPaths;
}

export interface Source {
  repo: string;
  branch: string;
  api_repo: string;
  api_files: string[];
}

export interface Config {
  source: Source;
  servers: Server[];
}

const BOX_DEFAULTS: BoxPaths = {
  venv_link: "~/opt/mempalace/.venv",
  src_link: "~/opt/mempalace-src",
  api_link: "~/opt/mempalace-api",
  python: "~/opt/python-3.14.6-sqlite-3.53.4/bin/python3.14",
  expect_python: "3.14.6",
  expect_sqlite: "3.53.4",
  api_unit: "mempalace-api.service",
  api_port: 4109,
  keep: 3,
  extra_packages: ["fastapi>=0.115", "uvicorn[standard]>=0.34"],
};

const LOCAL_DEFAULTS: LocalPaths = {
  mempalace_bin: "",
  python: "",
  api_port: 4109,
  backup_dir: "",
  backup_glob: "",
  backup_log: "",
};

// Paths and unit names travel to the box inside a shell header, so they are
// held to a plain character set rather than quoted cleverly.
const SAFE = /^[~A-Za-z0-9_.@\/:+=<>\[\]-]+$/;

function str(v: unknown, where: string, fallback?: string): string {
  if (v === undefined && fallback !== undefined) return fallback;
  if (typeof v !== "string") throw new Error(`${where}: expected a string`);
  return v;
}

function safe(v: string, where: string): string {
  if (v !== "" && !SAFE.test(v)) throw new Error(`${where}: '${v}' has characters fleet will not ship to a shell`);
  return v;
}

// check=false for local paths: they go to argv, never through a shell.
function merge<T extends object>(defaults: T, given: unknown, where: string, check = true): T {
  const ok = (x: string, w: string) => (check ? safe(x, w) : x);
  const out = { ...defaults } as Record<string, unknown>;
  if (given === undefined) return out as T;
  if (typeof given !== "object" || given === null) throw new Error(`${where}: expected an object`);
  for (const [k, v] of Object.entries(given)) {
    if (!(k in defaults)) throw new Error(`${where}.${k}: unknown key`);
    const d = (defaults as Record<string, unknown>)[k];
    if (Array.isArray(d)) {
      if (!Array.isArray(v) || v.some((x) => typeof x !== "string"))
        throw new Error(`${where}.${k}: expected a list of strings`);
      out[k] = v.map((x: string) => ok(x, `${where}.${k}`));
    } else if (typeof d === "number") {
      if (typeof v !== "number" || !Number.isInteger(v) || v < 0) throw new Error(`${where}.${k}: expected a whole number`);
      out[k] = v;
    } else {
      out[k] = ok(str(v, `${where}.${k}`), `${where}.${k}`);
    }
  }
  return out as T;
}

export function parseConfig(text: string): Config {
  const raw = JSON.parse(text) as Record<string, unknown>;
  const src = (raw.source ?? {}) as Record<string, unknown>;
  const source: Source = {
    repo: str(src.repo, "source.repo"),
    branch: str(src.branch, "source.branch", "xdev-patches"),
    api_repo: str(src.api_repo, "source.api_repo"),
    api_files: Array.isArray(src.api_files)
      ? src.api_files.map((f, i) => safe(str(f, `source.api_files[${i}]`), "source.api_files"))
      : ["mempalace_api.py", "briefing_apply.py", "briefing_parser.py", "pyproject.toml"],
  };
  const boxDefaults = merge(BOX_DEFAULTS, raw.box_defaults, "box_defaults");
  if (!Array.isArray(raw.servers) || raw.servers.length === 0) throw new Error("servers: expected a non-empty list");
  const seen = new Set<string>();
  const servers = raw.servers.map((s: Record<string, unknown>, i: number): Server => {
    const where = `servers[${i}]`;
    const name = safe(str(s.name, `${where}.name`), `${where}.name`);
    if (seen.has(name)) throw new Error(`${where}.name: '${name}' is listed twice`);
    seen.add(name);
    const kind = str(s.kind, `${where}.kind`);
    if (kind !== "local" && kind !== "box") throw new Error(`${where}.kind: '${kind}' is not local or box`);
    if (typeof s.enabled !== "boolean") throw new Error(`${where}.enabled: expected true or false`);
    const ssh = safe(str(s.ssh, `${where}.ssh`, ""), `${where}.ssh`);
    if (kind === "box" && s.enabled && !ssh) throw new Error(`${where}.ssh: an enabled box needs an ssh target`);
    return {
      name,
      kind,
      enabled: s.enabled,
      note: str(s.note, `${where}.note`, ""),
      ssh,
      backup_timer: safe(str(s.backup_timer, `${where}.backup_timer`, ""), `${where}.backup_timer`),
      box: merge(boxDefaults, s.paths && kind === "box" ? s.paths : undefined, `${where}.paths`),
      local: merge(LOCAL_DEFAULTS, s.paths && kind === "local" ? s.paths : undefined, `${where}.paths`, false),
    };
  });
  return { source, servers };
}

export function selectServers(cfg: Config, target: string | undefined): Server[] {
  if (!target || target === "all") return cfg.servers;
  const s = cfg.servers.find((x) => x.name === target);
  if (!s) throw new Error(`no server '${target}' in the config (have: ${cfg.servers.map((x) => x.name).join(", ")})`);
  return [s];
}

// ── diff classifier ─────────────────────────────────────────────────────

// What ships in the installed wheel or changes what gets installed.
const RUNTIME = [/^mempalace\//, /^pyproject\.toml$/, /^setup\.(py|cfg)$/, /^MANIFEST\.in$/];

export type DiffClass = "same" | "runtime" | "non-runtime";

export function classifyPaths(paths: string[]): DiffClass {
  const files = paths.map((p) => p.trim()).filter(Boolean);
  if (files.length === 0) return "same";
  return files.some((f) => RUNTIME.some((r) => r.test(f))) ? "runtime" : "non-runtime";
}

export function describeDiff(behind: number, ahead: number, paths: string[]): string {
  if (behind === 0 && ahead === 0) return "current";
  const cls = classifyPaths(paths);
  const parts: string[] = [];
  if (behind > 0) parts.push(`${behind} behind`);
  if (ahead > 0) parts.push(`${ahead} not on branch`);
  const what = cls === "runtime" ? "RUNTIME" : cls === "non-runtime" ? "scripts/tests only" : "no file changes";
  return `${parts.join(", ")}: ${what}`;
}

// ── remote protocol ─────────────────────────────────────────────────────

export function parseKv(stdout: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of stdout.split("\n")) {
    const i = line.indexOf("=");
    if (i > 0) out[line.slice(0, i)] = line.slice(i + 1);
  }
  return out;
}

export function shq(v: string): string {
  return `'${v.replace(/'/g, `'\\''`)}'`;
}

// A config path as a shell word: ~/x becomes "$HOME"/'x' so the box expands it.
export function shPath(p: string): string {
  if (p === "~") return `"$HOME"`;
  if (p.startsWith("~/")) return `"$HOME"/${shq(p.slice(2))}`;
  return shq(p);
}

export function remoteHeader(vars: Record<string, string>, paths: Record<string, string> = {}): string {
  const lines = Object.entries(vars).map(([k, v]) => `${k}=${shq(v)}`);
  for (const [k, v] of Object.entries(paths)) lines.push(`${k}=${shPath(v)}`);
  return lines.join("\n") + "\n";
}

// ── rendering ───────────────────────────────────────────────────────────

export function renderTable(headers: string[], rows: string[][]): string {
  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => (r[i] ?? "").length)));
  const fmt = (r: string[]) =>
    r
      .map((c, i) => (c ?? "").padEnd(widths[i]))
      .join("  ")
      .trimEnd();
  return [fmt(headers), fmt(widths.map((w) => "-".repeat(w))), ...rows.map(fmt)].join("\n");
}

export function ago(from: Date, now: Date = new Date()): string {
  const s = Math.round((now.getTime() - from.getTime()) / 1000);
  if (!Number.isFinite(s)) return "?";
  if (s < 90) return `${s}s ago`;
  if (s < 5400) return `${Math.round(s / 60)}m ago`;
  if (s < 172800) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}
