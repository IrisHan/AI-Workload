import { useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import {
  Activity,
  ArrowLeft,
  Box,
  Boxes,
  ChevronRight,
  CircleHelp,
  Cpu,
  FlaskConical,
  Gauge,
  GitCompareArrows,
  Layers3,
  Menu,
  Network,
  PanelLeft,
  PanelLeftClose,
  Rocket,
  Settings2,
  X,
} from "lucide-react";
import { useHealth } from "../features/queries";

const NAV_STORAGE = "planner.shell.nav";

const groups = [
  {
    name: "Workspace",
    items: [
      { to: "/dashboard", label: "Overview", icon: Gauge },
      { to: "/models", label: "Model Center", icon: Box },
      { to: "/workloads", label: "Workload Center", icon: Layers3 },
      { to: "/hardware", label: "Hardware Center", icon: Cpu },
    ],
  },
  {
    name: "Planning",
    items: [
      { to: "/planning/roofline", label: "Roofline Analyzer", icon: Activity },
      { to: "/planning/resources", label: "Resource Planner", icon: Boxes },
      { to: "/planning/topology", label: "Topology Planner", icon: Network },
      {
        to: "/planning/compare",
        label: "Scenario Compare",
        icon: GitCompareArrows,
      },
      {
        to: "/planning/deployments",
        label: "Deployment Planner",
        icon: Rocket,
      },
    ],
  },
  {
    name: "Operations",
    items: [
      { to: "/runtime", label: "Runtime & Benchmarks", icon: FlaskConical },
      { to: "/administration", label: "Data & Versions", icon: Settings2 },
    ],
  },
];

function readNavCollapsed(): boolean {
  try {
    return window.localStorage.getItem(NAV_STORAGE) === "collapsed";
  } catch {
    return false;
  }
}

function writeNavCollapsed(collapsed: boolean) {
  try {
    window.localStorage.setItem(NAV_STORAGE, collapsed ? "collapsed" : "expanded");
  } catch {
    /* private mode */
  }
}

export function Layout() {
  const [open, setOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(readNavCollapsed);
  const health = useHealth();
  const location = useLocation();
  const studioMatch = location.pathname.match(/^\/models\/([^/]+)\/studio\/?$/);
  const studioFocus = Boolean(studioMatch);
  const structureTab =
    /^\/models\/[^/]+\/?$/.test(location.pathname) &&
    new URLSearchParams(location.search).get("tab") === "structure";
  const workloadCanvas = location.pathname.startsWith("/workloads/inference/");
  const wideCanvas = studioFocus || structureTab || workloadCanvas;
  const active = groups
    .flatMap((g) => g.items)
    .find((i) => location.pathname.startsWith(i.to));
  const shellClass = [
    "app-shell",
    collapsed && !studioFocus ? "nav-collapsed" : "",
    studioFocus ? "chrome-focus" : "",
    wideCanvas ? "wide-canvas" : "",
    workloadCanvas ? "workload-canvas" : "",
  ]
    .filter(Boolean)
    .join(" ");

  function toggleCollapsed() {
    setCollapsed((current) => {
      const next = !current;
      writeNavCollapsed(next);
      return next;
    });
  }

  return (
    <div className={shellClass}>
      {open && (
        <button
          className="nav-scrim"
          aria-label="Close navigation"
          onClick={() => setOpen(false)}
        />
      )}
      <aside id="app-sidebar" className={`sidebar ${open ? "open" : ""}`}>
        <div className="brand">
          <Boxes size={26} strokeWidth={1.5} />
          <span>
            AI Workload
            <br />
            <strong>Planner</strong>
          </span>
          <button
            className="icon-button mobile-only"
            title="Close navigation"
            onClick={() => setOpen(false)}
          >
            <X size={18} />
          </button>
        </div>
        <nav aria-label="Main navigation">
          {groups.map((group) => (
            <section className="nav-group" key={group.name}>
              <h2>{group.name}</h2>
              {group.items.map((item) => (
                <NavLink
                  to={item.to}
                  key={item.to}
                  title={item.label}
                  onClick={() => setOpen(false)}
                >
                  <item.icon size={18} />
                  <span>{item.label}</span>
                </NavLink>
              ))}
            </section>
          ))}
        </nav>
        <a className="docs-link" href="/docs" target="_blank" rel="noreferrer">
          <CircleHelp size={17} />
          <span>API Reference</span>
        </a>
      </aside>
      <div className="workspace">
        <header className="topbar">
          {studioFocus ? (
            <div className="breadcrumb">
              <Link
                className="back-link"
                to={`/models/${studioMatch![1]}${location.search}`}
              >
                <ArrowLeft size={16} />
                返回模型详情
              </Link>
              <strong>结构独立页</strong>
            </div>
          ) : (
            <div className="breadcrumb">
              <button
                className="icon-button mobile-only"
                title="Open navigation"
                onClick={() => setOpen(true)}
              >
                <Menu size={20} />
              </button>
              <button
                className="icon-button nav-toggle"
                title={collapsed ? "展开导航" : "收起导航"}
                aria-label={collapsed ? "展开导航" : "收起导航"}
                aria-expanded={!collapsed}
                aria-controls="app-sidebar"
                onClick={toggleCollapsed}
              >
                {collapsed ? <PanelLeft size={18} /> : <PanelLeftClose size={18} />}
              </button>
              <span>Workspace</span>
              <ChevronRight size={14} />
              <strong>{active?.label ?? "Planner"}</strong>
            </div>
          )}
          <span className={`connection ${health.isError ? "bad" : ""}`}>
            {health.isPending
              ? "Connecting"
              : health.isError
                ? "API unavailable"
                : "API connected"}
          </span>
        </header>
        <main>
          <Outlet />
        </main>
        <footer className="workspace-footer">
          <span>Analytical planning</span>
          <span>v0.1.0</span>
        </footer>
      </div>
    </div>
  );
}
