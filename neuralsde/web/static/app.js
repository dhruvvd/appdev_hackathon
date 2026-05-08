(() => {
  const $ = (id) => document.getElementById(id);

  const els = {
    run: $("run"),
    ticker: $("ticker"),
    epochs: $("epochs"),
    paths: $("paths"),
    status: $("status"),
    results: $("results"),
    resultsTitle: $("resultsTitle"),
    interpretRoot: $("interpretRoot"),
    metrics: $("metrics"),
    priceCaption: $("priceCaption"),
    lossCaption: $("lossCaption"),
    resultsDisclaimer: $("resultsDisclaimer"),
    priceCanvas: $("priceChart"),
    lossCanvas: $("lossChart"),
  };

  const usd = new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  });

  function formatUsd(n) {
    return usd.format(n);
  }

  function formatSignedPct(diff, base) {
    if (!Number.isFinite(base) || base === 0) return "—";
    const pct = (diff / base) * 100;
    const rounded = pct.toFixed(1);
    if (pct > 0) return `+${rounded}%`;
    return `${rounded}%`;
  }

  /**
   * @param {Record<string, unknown>} data
   */
  function renderInterpretation(data) {
    const ticker = String(data.ticker);
    const n = Number(data.n);
    const trainLen = Number(data.train_len);
    const testDays = n - trainLen;
    /** @type {number[]} */
    const actual = data.actual_usd;
    /** @type {number[]} */
    const mean = data.mean_usd;
    /** @type {number[]} */
    const lower = data.lower_usd;
    /** @type {number[]} */
    const upper = data.upper_usd;

    const last = n - 1;
    const lastActual = actual[last];
    const lastMean = mean[last];
    const lastLow = lower[last];
    const lastHigh = upper[last];
    const diffEnd = lastMean - lastActual;
    const bandWidth = lastHigh - lastLow;

    let maeTest = 0;
    if (testDays > 0) {
      for (let i = trainLen; i < n; i += 1) {
        maeTest += Math.abs(actual[i] - mean[i]);
      }
      maeTest /= testDays;
    }

    const ft = data.final_train_mse;
    const fv = data.final_val_mse;
    let fitClass = "tag-line";
    let fitText = "";
    if (typeof ft === "number" && typeof fv === "number" && ft > 0) {
      const ratio = fv / ft;
      if (ratio < 1.15) {
        fitText =
          "The model’s error on hidden days is only a little higher than on days it learned from—often a healthy sign that it is not only memorizing the past.";
      } else if (ratio < 2.8) {
        fitText =
          "The model fits history more easily than brand-new days, which is normal: the future is harder than the past.";
      } else {
        fitClass += " warn";
        fitText =
          "The gap between “practice” error and “hidden days” error is large. Treat the blue curve and shaded band as a loose illustration, not a precise forecast.";
      }
    } else {
      fitText =
        "Use the charts together: if the blue line drifts far from real prices after the dashed line, the model is struggling on unseen data.";
    }

    els.resultsTitle.textContent = `Results for ${ticker}`;

    els.interpretRoot.innerHTML = `
      <p class="interpret-lede">
        Here’s a plain-language read of what just ran. The model learns patterns from older closing prices, then we compare its simulations to real prices—including days it never trained on.
      </p>
      <div class="interpret-grid">
        <div class="interpret-card">
          <h4>Latest close in this dataset</h4>
          <span class="big-number">${formatUsd(lastActual)}</span>
          <p>
            The gray dashed divider marks day ${trainLen}: everything to the left is history the model was trained on; the ${testDays}-day segment on the right is “quiz” data shown in red.
          </p>
        </div>
        <div class="interpret-card">
          <h4>Typical simulated ending price</h4>
          <span class="big-number">${formatUsd(lastMean)}</span>
          <p>
            Versus that latest close, the average simulated path ends ${formatSignedPct(diffEnd, lastActual)}, or about ${formatUsd(Math.abs(diffEnd))} ${diffEnd >= 0 ? "higher" : "lower"}.
          </p>
        </div>
        <div class="interpret-card">
          <h4>How wide is the uncertainty band?</h4>
          <span class="big-number">± ~${formatUsd(bandWidth / 2)}</span>
          <p>
            The shaded band spans about ${formatUsd(lastLow)} to ${formatUsd(lastHigh)} on the last day. It shows disagreement across simulations—not a promise about tomorrow’s market range.
          </p>
        </div>
        <div class="interpret-card">
          <h4>Fit on hidden days</h4>
          <span class="big-number">${testDays > 0 ? formatUsd(maeTest) : "—"} avg gap</span>
          <p>
            On held-out days, the blue line is about this far from the red line <em>on average</em> (absolute dollar gap per day).
          </p>
        </div>
      </div>
      <ul class="interpret-list">
        <li><strong>Muted line:</strong> real closes during training.</li>
        <li><strong>Red line:</strong> real closes after the split—the model did not train on these.</li>
        <li><strong>Blue line:</strong> the average of many noisy simulations from the learned dynamics.</li>
        <li><strong>Blue band:</strong> spread across simulations at each day (rough “many futures” fan), not a formal confidence interval.</li>
      </ul>
      <p class="${fitClass}">${fitText}</p>
    `;

    const approxYears = (n / 252).toFixed(1);
    els.priceCaption.textContent = `Each point is one trading day (about ${n} days loaded here—roughly ${approxYears} years at ~252 trading days per year). The vertical dashed line is the train/test split: left = learned from, right = held out for comparison.`;

    els.lossCaption.textContent =
      'Orange tracks error on held-out days; blue tracks error on training days. Both shrink when learning goes well. If orange stays much higher than blue, forecasts beyond the dashed line should be taken skeptically.';

    els.resultsDisclaimer.textContent =
      "This page is for learning and experimentation only. Markets involve risks this toy model does not capture; it is not financial, legal, or tax advice.";
  }

  /** @type {import("chart.js").Chart | null} */
  let priceChart = null;
  /** @type {import("chart.js").Chart | null} */
  let lossChart = null;

  const chartFont = { family: "'DM Sans', system-ui, sans-serif" };

  function setStatus(msg, isError = false) {
    els.status.textContent = msg;
    els.status.classList.toggle("error", isError);
  }

  function destroyCharts() {
    if (priceChart) {
      priceChart.destroy();
      priceChart = null;
    }
    if (lossChart) {
      lossChart.destroy();
      lossChart = null;
    }
  }

  function renderMetrics(data) {
    const train = data.train_len;
    const test = data.n - train;
    const items = [
      ["Symbol", data.ticker],
      ["Trading days in chart", String(data.n)],
      ["Train days / test days", `${train} learned · ${test} hidden`],
      ["Training epochs run", String(data.num_epochs)],
      ["Simulation paths averaged", String(data.num_paths)],
      ["Final error on train (scaled)", data.final_train_mse != null ? data.final_train_mse.toFixed(6) : "—"],
      ["Final error on hidden tail (scaled)", data.final_val_mse != null ? data.final_val_mse.toFixed(6) : "—"],
    ];
    els.metrics.innerHTML = items
      .map(
        ([k, v]) =>
          `<div class="metric"><span>${k}</span><strong>${v}</strong></div>`
      )
      .join("");
  }

  /**
   * @param {number[]} xs
   * @param {number[]} ys
   */
  function trainTestActualDataset(xs, ys, trainLen) {
    const trainSeg = ys.map((y, i) => (i < trainLen ? y : null));
    const testSeg = ys.map((y, i) => (i >= trainLen ? y : null));
    return [
      {
        label: "Actual prices (training period)",
        data: xs.map((x, i) => ({ x, y: trainSeg[i] })),
        borderColor: "#94a3b8",
        backgroundColor: "#94a3b8",
        pointRadius: 0,
        tension: 0.1,
        spanGaps: false,
      },
      {
        label: "Actual prices (hidden / quiz period)",
        data: xs.map((x, i) => ({ x, y: testSeg[i] })),
        borderColor: "#f87171",
        backgroundColor: "#f87171",
        pointRadius: 0,
        tension: 0.1,
        spanGaps: false,
      },
    ];
  }

  function renderPriceChart(data) {
    const xs = data.day_index;
    const trainLen = data.train_len;
    const actual = data.actual_usd;
    const mean = data.mean_usd;
    const lower = data.lower_usd;
    const upper = data.upper_usd;

    const bandUpper = xs.map((x, i) => ({ x, y: upper[i] }));
    const bandLower = xs.map((x, i) => ({ x, y: lower[i] }));

    destroyCharts();

    priceChart = new Chart(els.priceCanvas.getContext("2d"), {
      type: "line",
      data: {
        datasets: [
          {
            label: "_band_lower",
            data: bandLower,
            borderColor: "transparent",
            pointRadius: 0,
            parsing: false,
          },
          {
            label: "Uncertainty band (spread of simulations)",
            data: bandUpper,
            borderColor: "transparent",
            backgroundColor: "rgba(96, 165, 250, 0.18)",
            fill: "-1",
            pointRadius: 0,
            parsing: false,
          },
          ...trainTestActualDataset(xs, actual, trainLen),
          {
            label: "Mean path (average simulation)",
            data: xs.map((x, i) => ({ x, y: mean[i] })),
            borderColor: "#60a5fa",
            backgroundColor: "#60a5fa",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.15,
            parsing: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            labels: {
              color: "#c9d6e8",
              font: chartFont,
              filter(item) {
                return item.text !== "_band_lower";
              },
            },
          },
          title: {
            display: true,
            text: `${data.ticker} — closing price (USD) with model paths`,
            color: "#e8eef6",
            font: { ...chartFont, size: 14 },
          },
          tooltip: {
            callbacks: {
              label(ctx) {
                const v = ctx.parsed.y;
                if (v == null) return `${ctx.dataset.label}: —`;
                return `${ctx.dataset.label}: ${v.toFixed(4)}`;
              },
            },
          },
        },
        scales: {
          x: {
            type: "linear",
            title: { display: true, text: "Trading day index", color: "#8b9cb3" },
            ticks: { color: "#8b9cb3" },
            grid: { color: "rgba(255,255,255,0.06)" },
          },
          y: {
            title: { display: true, text: "Price (USD)", color: "#8b9cb3" },
            ticks: { color: "#8b9cb3" },
            grid: { color: "rgba(255,255,255,0.06)" },
          },
        },
      },
      plugins: [
        {
          id: "trainSplitLine",
          afterDraw(chart) {
            const { ctx, chartArea, scales } = chart;
            const xScale = scales.x;
            const x = xScale.getPixelForValue(trainLen - 0.5);
            if (x < chartArea.left || x > chartArea.right) return;
            ctx.save();
            ctx.strokeStyle = "rgba(148, 163, 184, 0.6)";
            ctx.setLineDash([6, 6]);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, chartArea.top);
            ctx.lineTo(x, chartArea.bottom);
            ctx.stroke();
            ctx.restore();
          },
        },
      ],
    });

    const epochs = data.train_losses.map((_, i) => i + 1);
    lossChart = new Chart(els.lossCanvas.getContext("2d"), {
      type: "line",
      data: {
        labels: epochs,
        datasets: [
          {
            label: "Error on training days",
            data: data.train_losses,
            borderColor: "#60a5fa",
            backgroundColor: "transparent",
            pointRadius: 0,
            tension: 0.2,
          },
          {
            label: "Error on hidden days",
            data: data.val_losses,
            borderColor: "#fb923c",
            backgroundColor: "transparent",
            pointRadius: 0,
            tension: 0.2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#c9d6e8", font: chartFont } },
          title: {
            display: true,
            text: "Training quality (lower is better)",
            color: "#e8eef6",
            font: { ...chartFont, size: 14 },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "Epoch", color: "#8b9cb3" },
            ticks: { color: "#8b9cb3", maxTicksLimit: 12 },
            grid: { color: "rgba(255,255,255,0.06)" },
          },
          y: {
            title: { display: true, text: "Average squared error (scaled)", color: "#8b9cb3" },
            ticks: { color: "#8b9cb3" },
            grid: { color: "rgba(255,255,255,0.06)" },
          },
        },
      },
    });
  }

  async function run() {
    const ticker = els.ticker.value.trim().toUpperCase() || "SPY";
    const num_epochs = Number(els.epochs.value);
    const num_paths = Number(els.paths.value);

    els.run.disabled = true;
    destroyCharts();
    els.results.classList.add("hidden");
    setStatus("Training… this usually takes 30–90s depending on hardware.");

    try {
      const res = await fetch("/api/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, num_epochs, num_paths }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = body.detail || res.statusText || "Request failed";
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }

      renderInterpretation(body);
      renderMetrics(body);
      renderPriceChart(body);
      els.results.classList.remove("hidden");
      setStatus("Done.");
    } catch (e) {
      setStatus(e instanceof Error ? e.message : String(e), true);
    } finally {
      els.run.disabled = false;
    }
  }

  els.run.addEventListener("click", run);
})();
