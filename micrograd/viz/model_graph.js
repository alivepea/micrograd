import cytoscape from "https://esm.sh/cytoscape@3.30.4";

function colors(dark) {
  if (dark) {
    return {
      label: "#fafaf9",
      input: "#44403c",
      relu: "#115e59",
      linear: "#9a3412",
      loss: "#1e3a8a",
      stage: "#1c1917",
    };
  }
  return {
    label: "#1c1917",
    input: "#e7e5e4",
    relu: "#99f6e4",
    linear: "#fed7aa",
    loss: "#bfdbfe",
    stage: "#fafaf9",
  };
}

function stylesheet(dark) {
  const ink = colors(dark);
  return [
    {
      selector: "node",
      style: {
        label: "data(label)",
        "text-wrap": "wrap",
        "text-valign": "center",
        "text-halign": "center",
        "font-size": 11,
        "font-family": "ui-sans-serif, system-ui, sans-serif",
        color: ink.label,
        "background-color": ink.input,
        shape: "round-rectangle",
        width: 78,
        height: 36,
        "border-width": "data(border_width)",
        "border-color": "data(border_color)",
      },
    },
    {
      selector: 'node[kind = "relu"]',
      style: { "background-color": ink.relu },
    },
    {
      selector: 'node[kind = "linear"]',
      style: { "background-color": ink.linear },
    },
    {
      selector: 'node[kind = "loss"]',
      style: { "background-color": ink.loss },
    },
    {
      selector: "edge",
      style: {
        width: "data(width)",
        "line-color": "data(color)",
        "curve-style": "straight",
        opacity: 0.9,
      },
    },
    {
      selector: 'edge[kind = "flow"]',
      style: {
        "line-style": "dashed",
        "target-arrow-shape": "triangle",
        "target-arrow-color": "data(color)",
        "arrow-scale": 0.8,
      },
    },
    {
      selector: 'edge[kind = "grad"]',
      style: {
        "target-arrow-shape": "triangle",
        "target-arrow-color": "data(color)",
        "arrow-scale": 0.7,
      },
    },
    {
      selector: ".dim",
      style: { opacity: 0.08 },
    },
    {
      selector: "edge.dim",
      style: { opacity: 0, events: "no" },
    },
    {
      selector: "node.hot",
      style: { "border-width": 3, "border-color": "#0f766e" },
    },
    {
      selector: "edge.hot",
      style: { opacity: 1, width: 3.5 },
    },
  ];
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function addRow(table, cells, header) {
  const tr = document.createElement("tr");
  for (const text of cells) {
    const cell = document.createElement(header ? "th" : "td");
    cell.textContent = text;
    tr.appendChild(cell);
  }
  table.appendChild(tr);
}

function showDetail(panel, detail) {
  clear(panel);
  const title = document.createElement("h3");
  const body = detail;
  title.textContent = body.title;
  panel.appendChild(title);

  if (body.rows && body.rows.length) {
    const dl = document.createElement("dl");
    dl.className = "mg-kv";
    for (const [key, value] of body.rows) {
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = key;
      dd.textContent = value;
      dl.appendChild(dt);
      dl.appendChild(dd);
    }
    panel.appendChild(dl);
  }

  if (body.terms && body.terms.length) {
    const table = document.createElement("table");
    const columns = body.columns || [
      ["src", "来自"],
      ["x", "x"],
      ["w", "w"],
      ["prod", "w·x"],
      ["w_grad", "∂w"],
    ];
    addRow(
      table,
      columns.map((column) => column[1]),
      true,
    );
    for (const term of body.terms) {
      addRow(
        table,
        columns.map((column) => term[column[0]] ?? ""),
        false,
      );
    }
    panel.appendChild(table);
  }

  if (body.note) {
    const note = document.createElement("p");
    note.textContent = body.note;
    panel.appendChild(note);
  }
}

function paintSummary(el, summary) {
  const data = summary || {};
  let text = `x = (${data.x ?? ""})   y = ${data.y ?? ""}   score = ${data.score ?? ""}   hinge = ${data.hinge ?? ""}`;
  if (data.score_grad) text += `   ∂score = ${data.score_grad}`;
  el.textContent = text;
}

function legendHtml(mode) {
  if (mode === "backward") {
    return (
      '<span><i class="mg-swatch up"></i>正 ∂w</span>' +
      '<span><i class="mg-swatch down"></i>负 ∂w，越粗 |∂w| 越大</span>' +
      '<span><i class="mg-swatch zero"></i>零梯度</span>' +
      "<span>箭头从损失指回输入，节点里的数是梯度</span>"
    );
  }
  return (
    '<span><i class="mg-swatch pos"></i>正权重</span>' +
    '<span><i class="mg-swatch neg"></i>负权重，越粗 |w| 越大</span>' +
    '<span><i class="mg-box up"></i>正梯度</span>' +
    '<span><i class="mg-box down"></i>负梯度</span>' +
    "<span>节点里的数是前向值</span>"
  );
}

function emptyDetail(mode) {
  if (mode === "backward") {
    return {
      title: "梯度怎么流回去",
      rows: [],
      terms: [],
      note: "点一个神经元看梯度如何拆到每个权重，点一条边看 ∂w = x · ∂(w·x)。点空白处取消选择。",
    };
  }
  return {
    title: "权重、计算、梯度",
    rows: [],
    terms: [],
    note: "点一个神经元看 b + Σ w·x，点一条边看这个权重的梯度。点空白处取消选择。",
  };
}

// Cytoscape caches the container's viewport box. It refreshes that cache on
// resize, and on scroll of light-DOM ancestors. This widget sits in marimo's
// shadow root, so scrolling the notebook never reaches the cache and taps land
// away from the node under the pointer. Fullscreen resizes the output, which
// rebuilds the cache, so taps work there.
function trackContainerBox(cy, canvas) {
  const refresh = () => {
    cy.renderer().invalidateContainerClientCoordsCache();
  };
  const listen = (target, type) => {
    target.addEventListener(type, refresh, true);
    return () => target.removeEventListener(type, refresh, true);
  };
  const stops = [listen(canvas, "pointerdown"), listen(canvas, "pointermove")];
  let node = canvas;
  while (node) {
    stops.push(listen(node, "scroll"));
    if (node.parentNode) {
      node = node.parentNode;
    } else if (node.host) {
      node = node.host;
    } else {
      break;
    }
  }
  return () => {
    for (const stop of stops) stop();
  };
}

function render({ model, el }) {
  const wrap = document.createElement("div");
  wrap.className = "mg-wrap";
  const stage = document.createElement("div");
  stage.className = "mg-stage";
  const toolbar = document.createElement("div");
  toolbar.className = "mg-toolbar";
  const summary = document.createElement("div");
  summary.className = "mg-summary";
  const canvas = document.createElement("div");
  canvas.className = "mg-canvas";
  const legend = document.createElement("div");
  legend.className = "mg-legend";
  const detail = document.createElement("div");
  detail.className = "mg-detail";

  function button(text, onClick) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = text;
    btn.addEventListener("click", onClick);
    return btn;
  }

  toolbar.append(button("放大", () => zoomBy(1.25)), button("缩小", () => zoomBy(0.8)), button("适应", () => fit()), summary);
  stage.append(toolbar, canvas, legend);
  wrap.append(stage, detail);
  el.appendChild(wrap);

  function mode() {
    return model.get("mode") || "forward";
  }

  function paintChrome() {
    legend.innerHTML = legendHtml(mode());
    paintSummary(summary, model.get("summary"));
  }

  paintChrome();
  showDetail(detail, emptyDetail(mode()));

  let cy = null;
  let stopTrackingBox = () => {};
  const dark = () => window.matchMedia("(prefers-color-scheme: dark)").matches;

  function fit() {
    if (cy) cy.fit(undefined, 28);
  }

  function zoomBy(factor) {
    if (!cy) return;
    const rect = canvas.getBoundingClientRect();
    cy.zoom({
      level: cy.zoom() * factor,
      renderedPosition: { x: rect.width / 2, y: rect.height / 2 },
    });
  }

  function draw() {
    canvas.style.height = `${model.get("height") || 640}px`;
    stopTrackingBox();
    if (cy) cy.destroy();
    cy = cytoscape({
      container: canvas,
      elements: model.get("elements") || [],
      style: stylesheet(dark()),
      layout: { name: "preset", fit: true, padding: 28 },
      userZoomingEnabled: false,
      boxSelectionEnabled: false,
      autoungrabify: true,
      minZoom: 0.15,
      maxZoom: 3,
    });
    cy.on("tap", "node, edge", (event) => {
      const picked = event.target;
      cy.elements().removeClass("hot").addClass("dim");
      if (picked.isNode()) {
        const hood = picked.closedNeighborhood();
        hood.removeClass("dim");
        picked.addClass("hot");
        // Only this neuron's edges. Neighborhood edges also touch every
        // neighbor, which would light up the rest of the layer.
        picked.connectedEdges().addClass("hot");
      } else {
        picked.removeClass("dim").addClass("hot");
        picked.connectedNodes().removeClass("dim").addClass("hot");
      }
      showDetail(detail, picked.data("detail"));
    });
    cy.on("tap", (event) => {
      if (event.target !== cy) return;
      cy.elements().removeClass("dim hot");
      showDetail(detail, emptyDetail(mode()));
    });
    stopTrackingBox = trackContainerBox(cy, canvas);
  }

  draw();
  model.on("change:elements", draw);
  model.on("change:height", draw);
  model.on("change:mode", () => {
    paintChrome();
    showDetail(detail, emptyDetail(mode()));
  });
  model.on("change:summary", () => paintSummary(summary, model.get("summary")));

  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const onTheme = () => draw();
  media.addEventListener("change", onTheme);
  const observer = new ResizeObserver(() => {
    if (cy) cy.resize();
  });
  observer.observe(canvas);

  return () => {
    media.removeEventListener("change", onTheme);
    observer.disconnect();
    stopTrackingBox();
    if (cy) cy.destroy();
  };
}

export default { render };
