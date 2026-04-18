import { useRef, useEffect, useState, useCallback } from 'react';
import * as d3 from 'd3';

// Ray-casting point-in-polygon
function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];
    const intersect =
      yi > point[1] !== yj > point[1] &&
      point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

const MARGIN = { top: 12, right: 12, bottom: 12, left: 12 };

const PALETTE = [
  '#4e79a7', '#a0cbe8',
  '#f28e2b', '#ffbe7d',
  '#59a14f', '#8cd17d',
  '#b6992d', '#f1ce63',
  '#499894', '#86bcb6',
  '#e15759', '#ff9da7',
  '#79706e', '#bab0ac',
  '#d37295', '#fabfd2',
  '#b07aa1', '#d4a6c8',
  '#9d7660', '#d7b5a6',
];

function drawStar(ctx, cx, cy, outerR, innerR) {
  const spikes = 5;
  let rot = (Math.PI / 2) * 3;
  const step = Math.PI / spikes;
  ctx.beginPath();
  for (let i = 0; i < spikes; i++) {
    ctx.lineTo(cx + Math.cos(rot) * outerR, cy + Math.sin(rot) * outerR);
    rot += step;
    ctx.lineTo(cx + Math.cos(rot) * innerR, cy + Math.sin(rot) * innerR);
    rot += step;
  }
  ctx.closePath();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function ScatterPlot({
  data,
  selectedPoints,
  testPoints,
  dotSize,
  opacity,
  showLabels,
  labelFontSize,
  lassoMode,
  highlightedCluster,
  onSelection,
  onTooltip,
  resetZoomRef,
  phraseHighlightIndices,
  hoveredEntryIndex,
  clickedEntryIndex,
  searchMatchIndices,
  colorByClusters,
  selectionLabel,
  selectionColor,
  onPointClick,
  liveCoords,
  liveLabels,
  pointSimilarities,
  colorBySimilarity,
  pointMovements,
  colorByMovement,
  pinnedTrail,
  showColorBar = false,
  namedLabels = [],
  onRemoveLabel,
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [dims, setDims] = useState(null);
  const dimsRef = useRef(null);
  const dprRef = useRef(1);

  const transformRef = useRef(d3.zoomIdentity);
  const zoomRef = useRef(null);
  const pointDataRef = useRef([]);
  const xScaleRef = useRef(null);
  const yScaleRef = useRef(null);
  const uniqueLabelsRef = useRef([]);
  const simColorMapRef = useRef(null);
  const simRangeRef = useRef(null);   // [min, max] for similarity scale
  const moveRangeRef = useRef(null);  // [min, max] for movement scale
  const semanticLabelsRef = useRef(null);
  const pinnedPointRef = useRef(null);
  const lassoPathRef = useRef([]);
  const isDrawingRef = useRef(false);
  const rafRef = useRef(null);

  // Dot size animation
  const currentDotSizeRef = useRef(dotSize);
  const targetDotSizeRef = useRef(dotSize);
  const dotSizeAnimRef = useRef(null);

  // Stable callback refs
  const onSelectionRef = useRef(onSelection);
  const onTooltipRef = useRef(onTooltip);
  const onPointClickRef = useRef(onPointClick);
  const onRemoveLabelRef = useRef(onRemoveLabel);
  onSelectionRef.current = onSelection;
  onTooltipRef.current = onTooltip;
  onPointClickRef.current = onPointClick;
  onRemoveLabelRef.current = onRemoveLabel;

  const namedLabelsRef = useRef(namedLabels);
  namedLabelsRef.current = namedLabels;
  const namedLabelHitAreasRef = useRef([]);

  // Value refs — always current, never cause Effect re-runs
  const selectedPointsRef = useRef(selectedPoints);
  const opacityRef = useRef(opacity);
  const highlightedClusterRef = useRef(highlightedCluster);
  const phraseHighlightIndicesRef = useRef(phraseHighlightIndices);
  const hoveredEntryIndexRef = useRef(hoveredEntryIndex);
  const clickedEntryIndexRef = useRef(clickedEntryIndex);
  const searchMatchIndicesRef = useRef(searchMatchIndices);
  const selectionLabelRef = useRef(selectionLabel);
  const selectionColorRef = useRef(selectionColor);
  const colorByClustersRef = useRef(colorByClusters);
  const liveCoordsRef = useRef(liveCoords);
  const liveLabelsRef = useRef(liveLabels);
  const colorBySimilarityRef = useRef(colorBySimilarity);
  const pointMovementsRef = useRef(pointMovements);
  const colorByMovementRef = useRef(colorByMovement);
  const moveColorMapRef = useRef(null);
  const showLabelsRef = useRef(showLabels);
  const labelFontSizeRef = useRef(labelFontSize);
  const testPointsRef = useRef(testPoints);
  const pinnedTrailRef = useRef(pinnedTrail);
  const showColorBarRef = useRef(showColorBar);
  pinnedTrailRef.current = pinnedTrail;
  showColorBarRef.current = showColorBar;

  selectedPointsRef.current = selectedPoints;
  opacityRef.current = opacity;
  highlightedClusterRef.current = highlightedCluster;
  phraseHighlightIndicesRef.current = phraseHighlightIndices;
  hoveredEntryIndexRef.current = hoveredEntryIndex;
  clickedEntryIndexRef.current = clickedEntryIndex;
  searchMatchIndicesRef.current = searchMatchIndices;
  selectionLabelRef.current = selectionLabel;
  selectionColorRef.current = selectionColor;
  colorByClustersRef.current = colorByClusters;
  liveCoordsRef.current = liveCoords;
  liveLabelsRef.current = liveLabels;
  colorBySimilarityRef.current = colorBySimilarity;
  pointMovementsRef.current = pointMovements;
  colorByMovementRef.current = colorByMovement;
  showLabelsRef.current = showLabels;
  labelFontSizeRef.current = labelFontSize;
  testPointsRef.current = testPoints;
  dimsRef.current = dims;

  // ── Main draw function (reads all state from refs) ───────────────────────
  const drawScene = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const d = dimsRef.current;
    if (!d) return;
    const { width, height } = d;
    const dpr = dprRef.current;
    const innerW = width - MARGIN.left - MARGIN.right;
    const innerH = height - MARGIN.top - MARGIN.bottom;

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const points = pointDataRef.current;
    if (!points.length) return;

    const t = transformRef.current;
    const ds = currentDotSizeRef.current;
    const lCoords = liveCoordsRef.current;
    const lLabels = liveLabelsRef.current;
    const xScale = xScaleRef.current;
    const yScale = yScaleRef.current;
    const uniqueLabels = uniqueLabelsRef.current;
    const hlCluster = highlightedClusterRef.current;
    const hovIdx = hoveredEntryIndexRef.current;
    const clkIdx = clickedEntryIndexRef.current;
    const pinnedIdx = pinnedPointRef.current?.index ?? null;

    const selectedSet = new Set(selectedPointsRef.current || []);
    const phraseArr = phraseHighlightIndicesRef.current;
    const phraseSet = phraseArr?.length ? new Set(phraseArr) : null;
    const searchArr = searchMatchIndicesRef.current;
    const searchSet = searchArr?.length ? new Set(searchArr) : null;
    const hasSelection = selectedSet.size > 0;
    const hasPhrase = !!phraseSet;
    const hasSearch = !!searchSet;

    // Focused points drawn last (on top)
    const focused = new Set();
    if (hovIdx !== null && hovIdx !== undefined) focused.add(hovIdx);
    if (clkIdx !== null && clkIdx !== undefined) focused.add(clkIdx);
    if (pinnedIdx !== null) focused.add(pinnedIdx);

    const getPlotXY = (index) => {
      if (lCoords && lCoords[index]) return [xScale(lCoords[index][0]), yScale(lCoords[index][1])];
      const pd = points[index];
      return [pd.x, pd.y];
    };

    const toScreen = (px, py) => [
      MARGIN.left + t.x + px * t.k,
      MARGIN.top + t.y + py * t.k,
    ];

    const getColor = (index, label) => {
      if (selectionColorRef.current && hasSelection && selectedSet.has(index)) return selectionColorRef.current;
      if (colorBySimilarityRef.current && simColorMapRef.current) return simColorMapRef.current[index] ?? '#9ca3af';
      if (colorByMovementRef.current && moveColorMapRef.current) return moveColorMapRef.current[index] ?? '#9ca3af';
      if (!colorByClustersRef.current) return '#1e293b';
      if (label === -1) return '#9ca3af';
      return PALETTE[uniqueLabels.indexOf(label) % PALETTE.length];
    };

    const getAlpha = (index, label) => {
      if (hasPhrase) return phraseSet.has(index) ? 1 : 0.1;
      if (hasSearch) return searchSet.has(index) ? 1 : 0.1;
      if (hasSelection) return selectedSet.has(index) ? 1 : 0.15;
      if (hlCluster !== null && label !== hlCluster) return 0.08;
      return opacityRef.current;
    };

    // Clip drawing to plot area
    ctx.save();
    ctx.beginPath();
    ctx.rect(MARGIN.left, MARGIN.top, innerW, innerH);
    ctx.clip();
    ctx.translate(MARGIN.left + t.x, MARGIN.top + t.y);
    ctx.scale(t.k, t.k);

    // ── Pinned point trail ───────────────────────────────────────────────────
    const pTrail = pinnedTrailRef.current;
    if (pTrail && pTrail.length >= 2 && xScale && yScale) {
      ctx.strokeStyle = 'rgba(100, 100, 100, 0.35)';
      ctx.lineWidth = 1 / t.k;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(xScale(pTrail[0][0]), yScale(pTrail[0][1]));
      for (let i = 1; i < pTrail.length; i++) {
        ctx.lineTo(xScale(pTrail[i][0]), yScale(pTrail[i][1]));
      }
      ctx.stroke();
    }

    // Pass 1: normal points
    for (const pd of points) {
      if (focused.has(pd.index)) continue;
      const [px, py] = getPlotXY(pd.index);
      const label = lLabels ? lLabels[pd.index] : pd.label;
      const r = (hasPhrase && phraseSet.has(pd.index) ? ds * 2 : ds) / t.k;
      ctx.globalAlpha = getAlpha(pd.index, label);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = getColor(pd.index, label);
      ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,0.45)';
      ctx.lineWidth = 0.5 / t.k;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Pass 2: focused points on top
    for (const idx of focused) {
      const pd = points[idx];
      if (!pd) continue;
      const [px, py] = getPlotXY(idx);
      const label = lLabels ? lLabels[idx] : pd.label;
      const isPinned = idx === pinnedIdx;
      // Pinned (selected): slightly larger + thick dark outline
      // Hovered/clicked only: slightly larger, normal outline
      const r = isPinned ? (ds * 2) / t.k : (ds * 1.6) / t.k;
      ctx.globalAlpha = Math.max(opacityRef.current, 0.9);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = getColor(idx, label);
      ctx.fill();
      ctx.strokeStyle = isPinned ? '#111111' : 'rgba(0,0,0,0.6)';
      ctx.lineWidth = (isPinned ? 2 : 1) / t.k;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Test points (stars)
    const tp = testPointsRef.current;
    if (tp && tp.length && xScale && yScale) {
      for (const tpd of tp) {
        const px = xScale(tpd.coord[0]);
        const py = yScale(tpd.coord[1]);
        drawStar(ctx, px, py, 8 / t.k, 4 / t.k);
        ctx.globalAlpha = 0.95;
        ctx.fillStyle = '#FFD700';
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = '#FF4500';
        ctx.lineWidth = 1.5 / t.k;
        ctx.stroke();
      }
    }

    ctx.restore(); // pop clip + zoom transform

    // ── Cluster labels in screen space (collision-filtered) ─────────────────
    if (showLabelsRef.current && semanticLabelsRef.current && xScale) {
      const sLabels = semanticLabelsRef.current;
      const n = sLabels.length;
      const sumX = new Float64Array(n);
      const sumY = new Float64Array(n);
      const cnt = new Int32Array(n);

      for (const pd of points) {
        const label = lLabels ? lLabels[pd.index] : pd.label;
        if (label >= 0 && label < n) {
          const [px, py] = getPlotXY(pd.index);
          const [sx, sy] = toScreen(px, py);
          sumX[label] += sx;
          sumY[label] += sy;
          cnt[label]++;
        }
      }

      const fs = (labelFontSizeRef.current ?? 11) + 1;
      const padX = 7, padY = 3;
      ctx.font = `500 ${fs}px Inter, -apple-system, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      // Build candidates, sorted largest cluster first (highest priority)
      const candidates = [];
      for (let i = 0; i < n; i++) {
        if (!cnt[i]) continue;
        const cx = sumX[i] / cnt[i];
        const cy = sumY[i] / cnt[i] - ds - 4;
        if (cx < MARGIN.left || cx > width - MARGIN.right ||
            cy < MARGIN.top  || cy > height - MARGIN.bottom) continue;
        const tw = ctx.measureText(sLabels[i]).width;
        candidates.push({ i, cx, cy, tw, size: cnt[i] });
      }
      candidates.sort((a, b) => b.size - a.size);

      // Greedy collision filter — accept if bounding box doesn't overlap any accepted label
      // colMargin expands the collision rect beyond the drawn box to keep labels spaced apart
      const colMargin = 14;
      const accepted = [];
      for (const c of candidates) {
        const x0 = c.cx - c.tw / 2 - padX - colMargin;
        const x1 = c.cx + c.tw / 2 + padX + colMargin;
        const y0 = c.cy - fs / 2 - padY - colMargin;
        const y1 = c.cy + fs / 2 + padY + colMargin;
        let clash = false;
        for (const a of accepted) {
          if (x0 < a.x1 && x1 > a.x0 && y0 < a.y1 && y1 > a.y0) { clash = true; break; }
        }
        if (!clash) accepted.push({ ...c, x0, x1, y0, y1 });
      }

      for (const c of accepted) {
        const dimmed = hlCluster !== null && c.i !== hlCluster;
        ctx.globalAlpha = dimmed ? 0.08 : 1;

        ctx.fillStyle = 'rgba(255,255,255,0.97)';
        roundRect(ctx, c.cx - c.tw / 2 - padX, c.cy - fs / 2 - padY, c.tw + padX * 2, fs + padY * 2, (fs + padY * 2) / 2);
        ctx.fill();
        ctx.strokeStyle = 'rgba(15,23,42,0.18)';
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = '#0f172a';
        ctx.fillText(sLabels[c.i], c.cx, c.cy);
      }
      ctx.globalAlpha = 1;
    }

    // ── Selection label ──────────────────────────────────────────────────────
    const sLabel = selectionLabelRef.current;
    const sColor = selectionColorRef.current;
    const selPts = selectedPointsRef.current;
    if (sLabel && sColor && selPts?.length && xScale) {
      let sumXs = 0, sumYs = 0, count = 0;
      for (const idx of selPts) {
        if (!points[idx]) continue;
        const [px, py] = getPlotXY(idx);
        const [sx, sy] = toScreen(px, py);
        sumXs += sx;
        sumYs += sy;
        count++;
      }
      if (count > 0) {
        const cx = sumXs / count;
        const cy = sumYs / count - ds - 6;
        const fs = 11;
        ctx.font = `500 ${fs}px Inter, -apple-system, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const tw = ctx.measureText(sLabel).width;
        const padX = 7, padY = 3;
        ctx.globalAlpha = 0.92;
        ctx.fillStyle = sColor;
        roundRect(ctx, cx - tw / 2 - padX, cy - fs / 2 - padY, tw + padX * 2, fs + padY * 2, 5);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.fillStyle = '#ffffff';
        ctx.fillText(sLabel, cx, cy);
      }
    }

    // ── Named persistent labels (with × close buttons) ──────────────────────
    namedLabelHitAreasRef.current = [];
    const nLabels = namedLabelsRef.current;
    if (nLabels?.length && xScale) {
      const fs = 11;
      ctx.font = `500 ${fs}px Inter, -apple-system, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (const nl of nLabels) {
        let sumXs = 0, sumYs = 0, count = 0;
        for (const idx of nl.indices) {
          if (!points[idx]) continue;
          const [px, py] = getPlotXY(idx);
          const [sx, sy] = toScreen(px, py);
          sumXs += sx; sumYs += sy; count++;
        }
        if (!count) continue;
        const cx = sumXs / count;
        const cy = sumYs / count - ds - 6;
        if (cx < MARGIN.left || cx > width - MARGIN.right ||
            cy < MARGIN.top  || cy > height - MARGIN.bottom) continue;
        const tw = ctx.measureText(nl.label).width;
        const padX = 7, padY = 3;
        const bx = cx - tw / 2 - padX;
        const by = cy - fs / 2 - padY;
        const bw = tw + padX * 2;
        const bh = fs + padY * 2;
        // Badge
        ctx.globalAlpha = 0.92;
        ctx.fillStyle = nl.color;
        roundRect(ctx, bx, by, bw, bh, 5);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.font = `500 ${fs}px Inter, -apple-system, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#ffffff';
        ctx.fillText(nl.label, cx, cy);
        // Close button: small dark circle to the right of the badge
        const btnR = 7;
        const btnX = bx + bw + btnR + 2;
        const btnY = cy;
        ctx.globalAlpha = 0.8;
        ctx.beginPath();
        ctx.arc(btnX, btnY, btnR, 0, Math.PI * 2);
        ctx.fillStyle = '#1f2937';
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.fillStyle = '#ffffff';
        ctx.font = `500 10px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('×', btnX, btnY);
        namedLabelHitAreasRef.current.push({ id: nl.id, btnX, btnY, btnR });
      }
    }

    // ── Color bar (bottom-right) ─────────────────────────────────────────────
    if (showColorBarRef.current) {
      const cbR = width - MARGIN.right - 32;
      const cbB = height - MARGIN.bottom - 12;
      const isSim = colorBySimilarityRef.current && simColorMapRef.current;
      const isMov = colorByMovementRef.current && moveColorMapRef.current;

      if (isSim || isMov) {
        const range = isSim ? simRangeRef.current : moveRangeRef.current;
        const colorFn = isSim ? d3.interpolatePlasma : d3.interpolateYlOrRd;
        const title = isSim ? 'Similarity' : 'Movement';
        if (range) {
          const barW = 21, barH = 173, steps = 60;
          const barX = cbR - barW, barY = cbB - barH;

          // Gradient bar (vertical: top = max, bottom = min)
          const grad = ctx.createLinearGradient(0, barY, 0, barY + barH);
          for (let i = 0; i <= steps; i++) {
            grad.addColorStop(i / steps, colorFn(1 - i / steps));
          }
          ctx.save();
          ctx.globalAlpha = 0.92;
          ctx.fillStyle = grad;
          ctx.fillRect(barX, barY, barW, barH);
          ctx.strokeStyle = 'rgba(0,0,0,0.18)';
          ctx.lineWidth = 0.5;
          ctx.strokeRect(barX, barY, barW, barH);

          // Labels
          ctx.fillStyle = '#111';
          ctx.font = '500 11px Inter, sans-serif';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          const fmt = (v) => Math.abs(v) < 0.01 ? v.toExponential(1) : v.toFixed(2);
          ctx.fillText(fmt(range[1]), barX + barW + 6, barY);
          ctx.fillText(fmt(range[0]), barX + barW + 6, barY + barH);

          // Title (rotated, left of bar)
          ctx.save();
          ctx.translate(barX - 6, barY + barH / 2);
          ctx.rotate(-Math.PI / 2);
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.font = '500 11px Inter, sans-serif';
          ctx.fillText(title, 0, 0);
          ctx.restore();

          ctx.restore();
        }
      }
    }

    // ── Lasso path ────────────────────────────────────────────────────────────
    if (isDrawingRef.current && lassoPathRef.current.length > 1) {
      const lp = lassoPathRef.current;
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(lp[0][0], lp[0][1]);
      for (let i = 1; i < lp.length; i++) ctx.lineTo(lp[i][0], lp[i][1]);
      ctx.closePath();
      ctx.fillStyle = 'rgba(100,149,237,0.15)';
      ctx.fill();
      ctx.strokeStyle = 'cornflowerblue';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }
  }, []); // all reads via refs — stable forever

  const scheduleDraw = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      drawScene();
      rafRef.current = null;
    });
  }, [drawScene]);

  // ── ResizeObserver ────────────────────────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) setDims({ width, height });
      }
    });
    ro.observe(el);
    const { width, height } = el.getBoundingClientRect();
    if (width > 0 && height > 0) setDims({ width, height });
    return () => ro.disconnect();
  }, []);

  // ── Effect A: Build scene on data/dims change ──────────────────────────────
  useEffect(() => {
    if (!data || !dims) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const { width, height } = dims;
    const dpr = window.devicePixelRatio || 1;
    dprRef.current = dpr;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const innerW = width - MARGIN.left - MARGIN.right;
    const innerH = height - MARGIN.top - MARGIN.bottom;

    const xs = data.coords.map((c) => c[0]);
    const ys = data.coords.map((c) => c[1]);
    const [xMin, xMax] = d3.extent(xs);
    const [yMin, yMax] = d3.extent(ys);
    const xPad = (xMax - xMin) * 0.06;
    const yPad = (yMax - yMin) * 0.06;

    xScaleRef.current = d3.scaleLinear().domain([xMin - xPad, xMax + xPad]).range([0, innerW]);
    yScaleRef.current = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([innerH, 0]);

    uniqueLabelsRef.current = [...new Set(data.labels)].filter((l) => l !== -1).sort((a, b) => a - b);
    semanticLabelsRef.current = data.semantic_labels || null;

    pointDataRef.current = data.texts.map((text, i) => ({
      text,
      x: xScaleRef.current(data.coords[i][0]),
      y: yScaleRef.current(data.coords[i][1]),
      label: data.labels[i],
      index: i,
    }));

    currentDotSizeRef.current = dotSize;
    targetDotSizeRef.current = dotSize;

    // Create zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.3, 20])
      .on('zoom', (event) => {
        transformRef.current = event.transform;
        scheduleDraw();
      });
    zoomRef.current = zoom;

    if (resetZoomRef) {
      resetZoomRef.current = () => {
        transformRef.current = d3.zoomIdentity;
        d3.select(canvas).call(zoom.transform, d3.zoomIdentity);
      };
    }

    scheduleDraw();
  }, [data, dims]); // eslint-disable-line

  // ── Effect: Mouse events (re-runs on lassoMode or scene rebuild) ───────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const zoom = zoomRef.current;

    // Clear all previous handlers
    d3.select(canvas)
      .on('.zoom', null)
      .on('mousedown.lasso', null)
      .on('mousemove.lasso', null)
      .on('mouseup.lasso', null)
      .on('mousemove.hover', null)
      .on('mouseout.tooltip', null)
      .on('click.point', null);

    const getPointAt = (mx, my) => {
      const t = transformRef.current;
      // Convert canvas pixel coords → plot space
      const px = (mx - MARGIN.left - t.x) / t.k;
      const py = (my - MARGIN.top - t.y) / t.k;
      const ds = currentDotSizeRef.current;
      const hitR = Math.max(ds * 1.5, 8) / t.k;
      const threshold2 = hitR * hitR;
      const lCoords = liveCoordsRef.current;
      const xScale = xScaleRef.current;
      const yScale = yScaleRef.current;
      if (!xScale || !yScale) return null;
      let best = null, bestDist2 = threshold2;
      for (const pd of pointDataRef.current) {
        const dpx = lCoords && lCoords[pd.index] ? xScale(lCoords[pd.index][0]) : pd.x;
        const dpy = lCoords && lCoords[pd.index] ? yScale(lCoords[pd.index][1]) : pd.y;
        const d2 = (dpx - px) ** 2 + (dpy - py) ** 2;
        if (d2 < bestDist2) { bestDist2 = d2; best = pd; }
      }
      return best;
    };

    // Hover tooltip — always show on hover, always hide on leave
    d3.select(canvas).on('mousemove.hover', (event) => {
      const [mx, my] = d3.pointer(event, canvas);
      const hit = getPointAt(mx, my);
      if (hit) {
        const [bx, by] = d3.pointer(event, document.body);
        onTooltipRef.current({ visible: true, x: bx, y: by, text: hit.text });
        canvas.style.cursor = 'pointer';
      } else {
        onTooltipRef.current({ visible: false, x: 0, y: 0, text: '' });
        canvas.style.cursor = lassoMode ? 'crosshair' : 'grab';
      }
    });

    d3.select(canvas).on('mouseout.tooltip', () => {
      onTooltipRef.current({ visible: false, x: 0, y: 0, text: '' });
    });

    // Click to select point → populates right panel (no tooltip pinning)
    d3.select(canvas).on('click.point', (event) => {
      if (isDrawingRef.current) return;
      const [mx, my] = d3.pointer(event, canvas);
      // Named-label close button hit detection (checked before point hit)
      for (const area of namedLabelHitAreasRef.current) {
        const dx = mx - area.btnX, dy = my - area.btnY;
        if (dx * dx + dy * dy <= area.btnR * area.btnR) {
          event.stopPropagation();
          onRemoveLabelRef.current?.(area.id);
          return;
        }
      }
      const hit = getPointAt(mx, my);
      if (hit) {
        event.stopPropagation();
        if (pinnedPointRef.current?.index === hit.index) {
          pinnedPointRef.current = null;
          onPointClickRef.current?.(null);
        } else {
          pinnedPointRef.current = { index: hit.index };
          onPointClickRef.current?.(hit.index);
        }
        scheduleDraw();
      }
    });

    if (lassoMode) {
      canvas.style.cursor = 'crosshair';
      d3.select(canvas)
        .on('mousedown.lasso', (event) => {
          if (event.button !== 0) return;
          isDrawingRef.current = true;
          lassoPathRef.current = [d3.pointer(event, canvas)];
          scheduleDraw();
        })
        .on('mousemove.lasso', (event) => {
          if (!isDrawingRef.current) return;
          lassoPathRef.current.push(d3.pointer(event, canvas));
          scheduleDraw();
        })
        .on('mouseup.lasso', () => {
          if (!isDrawingRef.current) return;
          isDrawingRef.current = false;
          const polygon = [...lassoPathRef.current];
          lassoPathRef.current = [];
          if (polygon.length >= 3) {
            const t = transformRef.current;
            const lCoords = liveCoordsRef.current;
            const xScale = xScaleRef.current;
            const yScale = yScaleRef.current;
            const selected = [];
            for (const pd of pointDataRef.current) {
              const dpx = lCoords && lCoords[pd.index] ? xScale(lCoords[pd.index][0]) : pd.x;
              const dpy = lCoords && lCoords[pd.index] ? yScale(lCoords[pd.index][1]) : pd.y;
              const sx = MARGIN.left + t.x + dpx * t.k;
              const sy = MARGIN.top + t.y + dpy * t.k;
              if (pointInPolygon([sx, sy], polygon)) selected.push(pd.index);
            }
            if (pinnedPointRef.current) {
              pinnedPointRef.current = null;
              onPointClickRef.current?.(null);
            }
            onSelectionRef.current(selected);
          }
          scheduleDraw();
        });
    } else {
      canvas.style.cursor = 'grab';
      if (zoom) {
        d3.select(canvas).call(zoom).call(zoom.transform, transformRef.current);
      }
    }
  }, [lassoMode, data, dims, scheduleDraw]); // eslint-disable-line

  // ── Effect D: Live coord update ───────────────────────────────────────────
  useEffect(() => {
    if (!liveCoords?.length || !xScaleRef.current) return;
    pointDataRef.current.forEach((pd) => {
      const c = liveCoords[pd.index];
      if (c) { pd.x = xScaleRef.current(c[0]); pd.y = yScaleRef.current(c[1]); }
    });
    scheduleDraw();
  }, [liveCoords, scheduleDraw]);

  // ── Effect E: Live label update ───────────────────────────────────────────
  useEffect(() => {
    if (!liveLabels?.length) return;
    pointDataRef.current.forEach((pd) => {
      if (liveLabels[pd.index] !== undefined) pd.label = liveLabels[pd.index];
    });
    uniqueLabelsRef.current = [...new Set(liveLabels)].filter((l) => l !== -1).sort((a, b) => a - b);
    scheduleDraw();
  }, [liveLabels, scheduleDraw]);

  // ── Effect F: Similarity color map ────────────────────────────────────────
  useEffect(() => {
    if (!pointSimilarities?.length) {
      simColorMapRef.current = null;
      scheduleDraw();
      return;
    }
    const min = Math.min(...pointSimilarities);
    const max = Math.max(...pointSimilarities);
    simRangeRef.current = [min, max];
    const scale = d3.scaleSequential(d3.interpolatePlasma).domain([min, max]);
    simColorMapRef.current = pointSimilarities.map((s) => scale(s));
    scheduleDraw();
  }, [pointSimilarities, scheduleDraw]);

  // ── Effect: Redraw when pinned trail changes ───────────────────────────────
  useEffect(() => {
    scheduleDraw();
  }, [pinnedTrail, scheduleDraw]);

  // ── Effect G: Movement color map ──────────────────────────────────────────
  useEffect(() => {
    if (!pointMovements?.length) {
      moveColorMapRef.current = null;
      scheduleDraw();
      return;
    }
    const min = Math.min(...pointMovements);
    const max = Math.max(...pointMovements);
    moveRangeRef.current = [min, max];
    const scale = d3.scaleSequential(d3.interpolateYlOrRd).domain([min, max]);
    moveColorMapRef.current = pointMovements.map((v) => scale(v));
    scheduleDraw();
  }, [pointMovements, scheduleDraw]);

  // ── Effect B: Re-draw on prop changes ────────────────────────────────────
  useEffect(() => {
    scheduleDraw();
  }, [selectedPoints, opacity, highlightedCluster, phraseHighlightIndices, hoveredEntryIndex,
    clickedEntryIndex, searchMatchIndices, selectionColor, selectionLabel, colorBySimilarity,
    colorByMovement, showLabels, colorByClusters, namedLabels, scheduleDraw]);

  // ── Dot size animation ────────────────────────────────────────────────────
  useEffect(() => {
    targetDotSizeRef.current = dotSize;
    const animate = () => {
      const curr = currentDotSizeRef.current;
      const tgt = targetDotSizeRef.current;
      if (Math.abs(curr - tgt) < 0.1) {
        currentDotSizeRef.current = tgt;
        scheduleDraw();
        return;
      }
      currentDotSizeRef.current = curr + (tgt - curr) * 0.25;
      scheduleDraw();
      dotSizeAnimRef.current = requestAnimationFrame(animate);
    };
    if (dotSizeAnimRef.current) cancelAnimationFrame(dotSizeAnimRef.current);
    dotSizeAnimRef.current = requestAnimationFrame(animate);
    return () => { if (dotSizeAnimRef.current) cancelAnimationFrame(dotSizeAnimRef.current); };
  }, [dotSize, scheduleDraw]);

  return (
    <div ref={containerRef} className="scatter-plot-container">
      <canvas
        ref={canvasRef}
        style={{ display: 'block', visibility: dims ? 'visible' : 'hidden' }}
      />
    </div>
  );
}

export default ScatterPlot;
