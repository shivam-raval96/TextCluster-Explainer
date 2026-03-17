import React, { useRef, useEffect, useState } from 'react';
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
}) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const zoomRef = useRef(null);
  const transformRef = useRef(d3.zoomIdentity);
  const lassoPathRef = useRef([]);
  const isDrawingRef = useRef(false);
  // Stable refs so Effect A doesn't re-run when callbacks change identity
  const onSelectionRef = useRef(onSelection);
  const onTooltipRef = useRef(onTooltip);
  onSelectionRef.current = onSelection;
  onTooltipRef.current = onTooltip;
  // pointData is shared between Effect A (writer) and Effect B (reader)
  const pointDataRef = useRef([]);

  // Value refs — always current, let applyHighlight read them without being in Effect A deps
  const selectedPointsRef = useRef(selectedPoints);
  const opacityRef = useRef(opacity);
  const highlightedClusterRef = useRef(highlightedCluster);
  const phraseHighlightIndicesRef = useRef(phraseHighlightIndices);
  const dotSizeRef = useRef(dotSize);
  const hoveredEntryIndexRef = useRef(hoveredEntryIndex);
  const clickedEntryIndexRef = useRef(clickedEntryIndex);
  const searchMatchIndicesRef = useRef(searchMatchIndices);
  selectedPointsRef.current = selectedPoints;
  opacityRef.current = opacity;
  highlightedClusterRef.current = highlightedCluster;
  phraseHighlightIndicesRef.current = phraseHighlightIndices;
  dotSizeRef.current = dotSize;
  hoveredEntryIndexRef.current = hoveredEntryIndex;
  clickedEntryIndexRef.current = clickedEntryIndex;
  searchMatchIndicesRef.current = searchMatchIndices;

  // Shared highlight function — called at end of Effect A and in Effect B
  const applyHighlightRef = useRef(null);
  applyHighlightRef.current = () => {
    if (!svgRef.current) return;
    const phraseArr = phraseHighlightIndicesRef.current;
    const phraseSet = phraseArr?.length ? new Set(phraseArr) : null;
    const searchArr = searchMatchIndicesRef.current;
    const searchSet = searchArr?.length ? new Set(searchArr) : null;
    const selectedSet = new Set(selectedPointsRef.current || []);
    const hasPhrase = !!phraseSet;
    const hasSearch = !!searchSet;
    const hasSelection = selectedSet.size > 0;
    const ds = dotSizeRef.current;
    const hoveredIdx = hoveredEntryIndexRef.current;
    const clickedIdx = clickedEntryIndexRef.current;

    const circles = d3.select(svgRef.current).selectAll('circle.point');

    // Opacity and stroke-width update immediately (no transition needed)
    circles
      .attr('opacity', (d) => {
        if (hasPhrase) return phraseSet.has(d.index) ? 1 : 0.1;
        if (hasSearch) return searchSet.has(d.index) ? 1 : 0.1;
        if (hasSelection) return selectedSet.has(d.index) ? 1 : 0.15;
        if (highlightedClusterRef.current !== null && d.label !== highlightedClusterRef.current) return 0.08;
        return opacityRef.current;
      })
      .attr('stroke-width', (d) => {
        if (hasPhrase && phraseSet.has(d.index)) return 2;
        if (hasSelection && selectedSet.has(d.index)) return 1.5;
        if (clickedIdx !== null && d.index === clickedIdx) return 2;
        if (hoveredIdx !== null && d.index === hoveredIdx) return 1.5;
        return 0.5;
      });

    // Radius transitions smoothly
    circles
      .transition()
      .duration(250)
      .ease(d3.easeCubicOut)
      .attr('r', (d) => {
        if (hasPhrase && phraseSet.has(d.index)) return ds * 2;
        if (clickedIdx !== null && d.index === clickedIdx) return ds * 5;
        if (hoveredIdx !== null && d.index === hoveredIdx) return ds * 5;
        return ds;
      });

    d3.select(svgRef.current).selectAll('g.cluster-label-group')
      .attr('opacity', (d) => {
        if (highlightedClusterRef.current !== null) {
          return d.i === highlightedClusterRef.current ? 1 : 0.08;
        }
        return 0.62;
      });

    // Raise focused point to the top of the SVG stack
    const focusIdx = clickedIdx !== null ? clickedIdx : hoveredIdx;
    if (focusIdx !== null) {
      d3.select(svgRef.current).selectAll('circle.point')
        .filter((d) => d.index === focusIdx)
        .raise();
    }
  };

  const [dims, setDims] = useState(null);

  // Observe container size — initialize from getBoundingClientRect on first observe
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
    // Also read immediately so we don't wait for the first ResizeObserver callback
    const { width, height } = el.getBoundingClientRect();
    if (width > 0 && height > 0) setDims({ width, height });
    return () => ro.disconnect();
  }, []);

  // ─── Effect A: Full scene build ──────────────────────────────────────────
  // Runs only when layout/structure/style changes — NOT on selection changes.
  useEffect(() => {
    if (!data || !svgRef.current || !dims) return;

    const { width, height } = dims;
    const innerW = width - MARGIN.left - MARGIN.right;
    const innerH = height - MARGIN.top - MARGIN.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    svg.append('defs')
      .append('clipPath').attr('id', 'plot-clip')
      .append('rect').attr('width', innerW).attr('height', innerH);

    const root = svg.append('g')
      .attr('class', 'root')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // Scales
    const xs = data.coords.map((c) => c[0]);
    const ys = data.coords.map((c) => c[1]);
    const xExtent = d3.extent(xs);
    const yExtent = d3.extent(ys);
    const xPad = (xExtent[1] - xExtent[0]) * 0.08;
    const yPad = (yExtent[1] - yExtent[0]) * 0.08;

    const xScale = d3.scaleLinear()
      .domain([xExtent[0] - xPad, xExtent[1] + xPad])
      .range([0, innerW]);
    const yScale = d3.scaleLinear()
      .domain([yExtent[0] - yPad, yExtent[1] + yPad])
      .range([innerH, 0]);

    const uniqueLabels = [...new Set(data.labels)].filter((l) => l !== -1).sort((a, b) => a - b);
    const color = (label) => {
      if (label === -1) return '#9ca3af';
      return PALETTE[uniqueLabels.indexOf(label) % PALETTE.length];
    };

    // Background rect for mouse events
    root.append('rect')
      .attr('class', 'plot-bg')
      .attr('width', innerW).attr('height', innerH)
      .attr('fill', 'transparent');

    // Plot area (receives zoom transform) — always restore saved transform
    const plotG = root.append('g')
      .attr('class', 'plot-area')
      .attr('clip-path', 'url(#plot-clip)')
      .style('will-change', 'transform')
      .attr('transform', transformRef.current);

    const pointsG = plotG.append('g').attr('class', 'points');
    const testG   = plotG.append('g').attr('class', 'test-points');
    const labelsG = plotG.append('g').attr('class', 'cluster-labels');
    const lassoG  = root.append('g').attr('class', 'lasso-layer');

    // ── Points ──────────────────────────────────────────────────────────────
    const pointData = data.texts.map((text, i) => ({
      text,
      x: xScale(data.coords[i][0]),
      y: yScale(data.coords[i][1]),
      label: data.labels[i],
      index: i,
    }));
    pointDataRef.current = pointData;

    pointsG.selectAll('circle.point')
      .data(pointData)
      .join('circle')
      .attr('class', 'point')
      .attr('cx', (d) => d.x)
      .attr('cy', (d) => d.y)
      .attr('r', dotSize)
      .attr('fill', (d) => color(d.label))
      .attr('opacity', opacity)
      .attr('stroke', 'rgba(0,0,0,0.55)')
      .attr('stroke-width', 0.5)
      .on('mouseover', (event, d) => {
        const [px, py] = d3.pointer(event, document.body);
        onTooltipRef.current({ visible: true, x: px, y: py, text: d.text });
      })
      .on('mousemove', (event) => {
        const [px, py] = d3.pointer(event, document.body);
        onTooltipRef.current((prev) => ({ ...prev, x: px, y: py }));
      })
      .on('mouseout', () => {
        onTooltipRef.current({ visible: false, x: 0, y: 0, text: '' });
      });

    // ── Labels ──────────────────────────────────────────────────────────────
    if (showLabels && data.semantic_labels) {
      const fs = labelFontSize ?? 11;
      const clusterCount = data.semantic_labels.length;
      const centroids = Array.from({ length: clusterCount }, () => ({ xs: [], ys: [] }));
      pointData.forEach((d) => {
        if (d.label >= 0 && d.label < clusterCount) {
          centroids[d.label].xs.push(d.x);
          centroids[d.label].ys.push(d.y);
        }
      });

      const nodes = data.semantic_labels.map((label, i) => ({ label, i }));

      const labelGroups = labelsG.selectAll('g.cluster-label-group')
        .data(nodes)
        .join('g')
        .attr('class', 'cluster-label-group')
        .attr('transform', (d) => {
          const c = centroids[d.i];
          const cx = c.xs.length ? d3.mean(c.xs) : 0;
          const cy = c.ys.length ? d3.mean(c.ys) - dotSize - 4 : 0;
          return `translate(${cx},${cy})`;
        })
        .attr('pointer-events', 'none')
        .attr('opacity', 0.95);

      labelGroups.append('rect')
        .attr('class', 'cluster-label-bg')
        .attr('rx', 6)
        .attr('ry', 6)
        .attr('fill', 'rgba(255,255,255,0.88)')
        .attr('stroke', 'rgba(0,0,0,0.07)')
        .attr('stroke-width', 0.5);

      labelGroups.append('text')
        .attr('class', 'cluster-label')
        .attr('x', 0)
        .attr('y', 0)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#000000')
        .attr('font-size', `${fs + 1}px`)
        .attr('font-weight', '700')
        .attr('pointer-events', 'none')
        .text((d) => d.label);

      // Fit background rects to text bounding boxes
      labelGroups.each(function () {
        const textEl = d3.select(this).select('text').node();
        try {
          const bbox = textEl.getBBox();
          const px = 7, py = 3;
          d3.select(this).select('rect')
            .attr('x', bbox.x - px)
            .attr('y', bbox.y - py)
            .attr('width', bbox.width + px * 2)
            .attr('height', bbox.height + py * 2);
        } catch (e) { /* getBBox unavailable if not rendered */ }
      });
    }

    // ── Test points ─────────────────────────────────────────────────────────
    if (testPoints && testPoints.length > 0) {
      const starSymbol = d3.symbol().type(d3.symbolStar).size(120);
      testG.selectAll('path.test-point')
        .data(testPoints)
        .join('path')
        .attr('class', 'test-point')
        .attr('d', starSymbol)
        .attr('transform', (d) => `translate(${xScale(d.coord[0])},${yScale(d.coord[1])})`)
        .attr('fill', '#FFD700')
        .attr('stroke', '#FF4500')
        .attr('stroke-width', 1.5)
        .attr('opacity', 0.95)
        .on('mouseover', (event, d) => {
          const [px, py] = d3.pointer(event, document.body);
          onTooltipRef.current({ visible: true, x: px, y: py, text: `★ ${d.text}` });
        })
        .on('mousemove', (event) => {
          const [px, py] = d3.pointer(event, document.body);
          onTooltipRef.current((prev) => ({ ...prev, x: px, y: py }));
        })
        .on('mouseout', () => {
          onTooltipRef.current({ visible: false, x: 0, y: 0, text: '' });
        });
    }

    // ── Zoom ────────────────────────────────────────────────────────────────
    const zoom = d3.zoom()
      .scaleExtent([0.3, 20])
      .on('zoom', (event) => {
        transformRef.current = event.transform;
        plotG.attr('transform', event.transform);
      });

    zoomRef.current = zoom;

    // Expose reset zoom to parent via ref
    if (resetZoomRef) {
      resetZoomRef.current = () => {
        transformRef.current = d3.zoomIdentity;
        svg.call(zoom.transform, d3.zoomIdentity);
      };
    }

    // ── Lasso ───────────────────────────────────────────────────────────────
    function getMouseInRoot(event) {
      return d3.pointer(event, root.node());
    }

    const lassoPath = lassoG.append('path')
      .attr('class', 'lasso-path')
      .attr('fill', 'rgba(100,149,237,0.15)')
      .attr('stroke', 'cornflowerblue')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3')
      .attr('display', 'none');

    function lassoMousedown(event) {
      event.preventDefault();
      isDrawingRef.current = true;
      lassoPathRef.current = [getMouseInRoot(event)];
      lassoPath.attr('display', null).attr('d', null).classed('lasso-path--drawing', true);
    }

    function lassoMousemove(event) {
      if (!isDrawingRef.current) return;
      lassoPathRef.current.push(getMouseInRoot(event));
      const line = d3.line()(lassoPathRef.current);
      lassoPath.attr('d', line ? line + 'Z' : null);
    }

    function lassoMouseup() {
      if (!isDrawingRef.current) return;
      isDrawingRef.current = false;
      lassoPath.classed('lasso-path--drawing', false);
      const polygon = lassoPathRef.current;
      if (polygon.length < 3) {
        lassoPath.attr('display', 'none');
        lassoPathRef.current = [];
        return;
      }
      const t = transformRef.current;
      const selected = [];
      pointDataRef.current.forEach((d) => {
        const rx = t.x + d.x * t.k;
        const ry = t.y + d.y * t.k;
        if (pointInPolygon([rx, ry], polygon)) selected.push(d.index);
      });
      onSelectionRef.current(selected);
      // Brief pulse on selected circles
      d3.select(svgRef.current).selectAll('circle.point')
        .filter((d) => {
          const rx = t.x + d.x * t.k;
          const ry = t.y + d.y * t.k;
          return pointInPolygon([rx, ry], polygon);
        })
        .classed('point--pulse', true);
      setTimeout(() => {
        lassoPath.attr('display', 'none').classed('lasso-path--drawing', false);
        lassoPathRef.current = [];
        d3.select(svgRef.current).selectAll('circle.point--pulse').classed('point--pulse', false);
      }, 700);
    }

    // ── Mode binding ────────────────────────────────────────────────────────
    const bgRect = root.select('rect.plot-bg');

    if (lassoMode) {
      svg.on('.zoom', null);
      bgRect.style('cursor', 'crosshair')
        .on('mousedown', lassoMousedown)
        .on('mousemove', lassoMousemove)
        .on('mouseup', lassoMouseup);
      svg.on('mouseup.lasso', lassoMouseup);
    } else {
      bgRect.style('cursor', 'grab')
        .on('mousedown', null).on('mousemove', null).on('mouseup', null);
      svg.on('mouseup.lasso', null);
      svg.call(zoom).call(zoom.transform, transformRef.current);
    }

    // Re-apply highlight/selection state onto freshly drawn circles
    applyHighlightRef.current();

  }, [data, testPoints, dotSize, showLabels, labelFontSize, dims, lassoMode, resetZoomRef]);

  // ─── Effect B: Update selection & highlight appearance (no full redraw) ──
  useEffect(() => {
    applyHighlightRef.current?.();
  }, [selectedPoints, opacity, highlightedCluster, phraseHighlightIndices, hoveredEntryIndex, clickedEntryIndex, searchMatchIndices]);

  return (
    <div ref={containerRef} className="scatter-plot-container">
      <svg ref={svgRef} className="scatter-plot-svg" style={{ visibility: dims ? 'visible' : 'hidden' }} />
    </div>
  );
}

export default ScatterPlot;
