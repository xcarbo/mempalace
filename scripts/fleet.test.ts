// Unit tests for the pure parts of scripts/fleet. Run: bun test scripts/fleet.test.ts
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { classifyPaths, describeDiff, parseConfig, parseKv, remoteHeader, selectServers, shPath } from "./fleet_lib.ts";

const minimal = (servers: unknown[], extra: Record<string, unknown> = {}) =>
  JSON.stringify({ source: { repo: "/r", api_repo: "/a" }, servers, ...extra });

describe("parseConfig", () => {
  test("the example config parses, and box-b is listed but disabled", () => {
    const cfg = parseConfig(readFileSync(join(import.meta.dir, "fleet.servers.example.json"), "utf8"));
    const boxB = cfg.servers.find((s) => s.name === "box-b");
    expect(boxB?.enabled).toBe(false);
    expect(cfg.servers.filter((s) => s.kind === "local").length).toBe(1);
    expect(cfg.source.branch).toBe("xdev-patches");
  });

  test("box defaults fill in, and a server may override one path", () => {
    const cfg = parseConfig(
      minimal([{ name: "b", kind: "box", enabled: true, ssh: "xdev@b", paths: { api_port: 4110 } }]),
    );
    expect(cfg.servers[0].box.api_port).toBe(4110);
    expect(cfg.servers[0].box.venv_link).toBe("~/opt/mempalace/.venv");
    expect(cfg.servers[0].box.expect_sqlite).toBe("3.53.4");
  });

  test("box_defaults apply to every box", () => {
    const cfg = parseConfig(
      minimal([{ name: "b", kind: "box", enabled: true, ssh: "x@b" }], { box_defaults: { keep: 5 } }),
    );
    expect(cfg.servers[0].box.keep).toBe(5);
  });

  test("rejects an unknown kind, a duplicate name, an unknown key, and shell metacharacters", () => {
    expect(() => parseConfig(minimal([{ name: "b", kind: "vm", enabled: true }]))).toThrow(/not local or box/);
    expect(() =>
      parseConfig(
        minimal([
          { name: "b", kind: "box", enabled: false },
          { name: "b", kind: "box", enabled: false },
        ]),
      ),
    ).toThrow(/twice/);
    expect(() => parseConfig(minimal([{ name: "b", kind: "box", enabled: false, paths: { venv: "x" } }]))).toThrow(
      /unknown key/,
    );
    expect(() =>
      parseConfig(minimal([{ name: "b", kind: "box", enabled: false, paths: { venv_link: "~/x; rm -rf ~" } }])),
    ).toThrow(/characters/);
  });

  test("an enabled box needs ssh; a disabled one does not", () => {
    expect(() => parseConfig(minimal([{ name: "b", kind: "box", enabled: true }]))).toThrow(/ssh target/);
    expect(parseConfig(minimal([{ name: "b", kind: "box", enabled: false }])).servers[0].ssh).toBe("");
  });

  test("selectServers", () => {
    const cfg = parseConfig(
      minimal([
        { name: "a", kind: "local", enabled: true },
        { name: "b", kind: "box", enabled: false },
      ]),
    );
    expect(selectServers(cfg, "all").length).toBe(2);
    expect(selectServers(cfg, undefined).length).toBe(2);
    expect(selectServers(cfg, "b")[0].name).toBe("b");
    expect(() => selectServers(cfg, "c")).toThrow(/no server 'c'/);
  });
});

describe("classifyPaths", () => {
  test("package code and packaging metadata are runtime", () => {
    expect(classifyPaths(["scripts/seed_palace.py", "mempalace/searcher.py"])).toBe("runtime");
    expect(classifyPaths(["pyproject.toml"])).toBe("runtime");
  });
  test("scripts, tests and docs are not", () => {
    expect(classifyPaths(["scripts/seed_palace.py", "tests/test_x.py", "CLAUDE.md", "uv.lock"])).toBe("non-runtime");
    // a path that merely contains the package name is not the package
    expect(classifyPaths(["tests/mempalace/x.py", "docs/mempalace/y.md"])).toBe("non-runtime");
  });
  test("no paths is the same tree", () => {
    expect(classifyPaths(["", "  "])).toBe("same");
  });
  test("describeDiff", () => {
    expect(describeDiff(0, 0, [])).toBe("current");
    expect(describeDiff(3, 0, ["scripts/a.py"])).toBe("3 behind: scripts/tests only");
    expect(describeDiff(1, 2, ["mempalace/a.py"])).toBe("1 behind, 2 not on branch: RUNTIME");
  });
});

describe("remote protocol", () => {
  test("parseKv splits on the first = only", () => {
    expect(parseKv("a=1\nb=x=y\n\nnoise\napi_sha256:f.py=abc\n")).toEqual({ a: "1", b: "x=y", "api_sha256:f.py": "abc" });
  });
  test("paths expand $HOME on the box and values are single-quoted", () => {
    expect(shPath("~/opt/m")).toBe(`"$HOME"/'opt/m'`);
    expect(shPath("/abs")).toBe(`'/abs'`);
    expect(remoteHeader({ CMD: "probe", N: "it's" }, { V: "~/v" })).toBe(`CMD='probe'\nN='it'\\''s'\nV="$HOME"/'v'\n`);
  });
});
