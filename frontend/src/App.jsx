import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import axios from 'axios';

import useDataset from './hooks/useDataset';
import ScatterPlot from './components/ScatterPlot';
import Legend from './components/Legend';
import ExplanationPanel, { SelectedTextsAccordion } from './components/ExplanationPanel';
import PhraseCloud from './components/PhraseCloud';
import TestInputPanel from './components/TestInputPanel';
import DatasetSelector from './components/DatasetSelector';
import Tooltip from './components/Tooltip';
import SelectionsHistoryPanel from './components/SelectionsHistoryPanel';
import DisplaySettings from './components/DisplaySettings';
import LMLeftPanel from './components/LMLeftPanel';
import LMDisplaySettings from './components/LMDisplaySettings';
import LMNewConceptModal from './components/LMNewConceptModal';
import LMWorkerToast from './components/LMWorkerToast';
import useLatentManipulator from './hooks/useLatentManipulator';
import { findInterpolationParams, interpolateCoordinates } from './utils/lmInterpolation';

import './styles/App.css';

const SELECTION_COLORS = [
  '#e63946', '#e76f51', '#d62828', '#9b2226',
  '#6a0dad', '#7b2d8b', '#3a0ca3', '#023e8a',
  '#0077b6', '#00b4d8', '#0a9396', '#2a9d8f',
  '#606c38', '#283618', '#bc6c25', '#774936',
];

function randomSelectionColor() {
  return SELECTION_COLORS[Math.floor(Math.random() * SELECTION_COLORS.length)];
}

function App() {
  const [appMode, setAppMode] = useState('explain'); // 'explain' | 'latent'
  const [showLMModal, setShowLMModal] = useState(false);
  const lm = useLatentManipulator();

  const [lmPinnedIndex, setLmPinnedIndex] = useState(null);
  const [lmClickedPointIndex, setLmClickedPointIndex] = useState(null);
  const [lmHighlightedCluster, setLmHighlightedCluster] = useState(null);
  const [lmLassoMode, setLmLassoMode] = useState(true);
  const [lmLabelFontSize, setLmLabelFontSize] = useState(11);
  const [lmWhiteBg, setLmWhiteBg] = useState(false);
  const [lmOpacity, setLmOpacity] = useState(0.75);
  const [lmCurrentSelection, setLmCurrentSelection] = useState(null);
  const [lmExplanation, setLmExplanation] = useState('');
  const [lmClusterLabel, setLmClusterLabel] = useState('');
  const [lmSelectionColor, setLmSelectionColor] = useState(null);
  const [lmIsExplaining, setLmIsExplaining] = useState(false);
  const [lmExplainError, setLmExplainError] = useState(null);
  const [lmExplainSettings, setLmExplainSettings] = useState({ model: 'gpt-5-nano-2025-08-07', promptTemplate: '' });

  const lmSemanticLabels = useMemo(() => {
    const topics = lm.currentTopics;
    if (!topics || !Object.keys(topics).length) return lm.lmBaseScatterData?.semantic_labels ?? [];
    const maxId = Math.max(...Object.keys(topics).map(Number));
    return Array.from({ length: maxId + 1 }, (_, i) => topics[i] ?? `Cluster ${i}`);
  }, [lm.currentTopics, lm.lmBaseScatterData]);
  const lmPinnedTrail = useMemo(
    () => lm.showTrail ? lm.getPinnedTrail(lmPinnedIndex) : null,
    [lm.showTrail, lmPinnedIndex, lm.getPinnedTrail],
  );

  const handleLMSelection = useCallback((indices) => {
    if (!lm.texts.length || indices.length === 0) return;
    const selectedTexts = indices.map((i) => lm.texts[i]);
    setLmCurrentSelection({ indices, texts: selectedTexts });
    setLmExplanation('');
    setLmClusterLabel('');
    setLmSelectionColor(null);
    setLmExplainError(null);
  }, [lm.texts]);

  const handleLMCancelSelection = useCallback(() => {
    setLmCurrentSelection(null);
    setLmExplanation('');
    setLmClusterLabel('');
    setLmSelectionColor(null);
    setLmExplainError(null);
  }, []);

  const handleLMExplain = useCallback(async () => {
    if (!lmCurrentSelection || !lm.texts.length) return;
    setLmIsExplaining(true);
    setLmExplainError(null);
    const { indices, texts: selectedTexts } = lmCurrentSelection;
    const selectedSet = new Set(indices);
    const otherTexts = lm.texts.filter((_, i) => !selectedSet.has(i));
    const shuffled = otherTexts.sort(() => Math.random() - 0.5).slice(0, 40);
    try {
      const explainBody = {
        dataset: lm.selectedDataset,
        selected_texts: selectedTexts,
        all_texts_sample: shuffled,
        model: lmExplainSettings.model,
      };
      if (lmExplainSettings.promptTemplate?.trim()) {
        explainBody.prompt_template = lmExplainSettings.promptTemplate;
      }
      const res = await axios.post('/api/explain', explainBody);
      const lmLabelText = res.data.cluster_label || '';
      const lmColor = randomSelectionColor();
      setLmExplanation(res.data.explanation || '');
      setLmSelectionColor(lmColor);
      if (lmLabelText && lmCurrentSelection) {
        setNamedLabels((prev) => [...prev, {
          id: Date.now().toString(),
          dataset: currentDataset,
          indices: lmCurrentSelection.indices,
          label: lmLabelText,
          color: lmColor,
        }]);
      }
    } catch (err) {
      setLmExplainError(err.response?.data?.error || err.message || 'Failed to get explanation');
    } finally {
      setLmIsExplaining(false);
    }
  }, [lmCurrentSelection, lm.texts, lm.selectedDataset, lmExplainSettings]);

  const [currentDataset, setCurrentDataset] = useState('vispub');
  const { data: datasetData, loading: dataLoading, error: dataError, staticMode } = useDataset(currentDataset);

  // Named labels — persistent per-dataset labels shown on the scatter plot
  // Shape: [{ id, dataset, indices, label, color }]
  const [namedLabels, setNamedLabels] = useState(() => {
    try {
      const stored = localStorage.getItem('textcluster_named_labels');
      return stored ? JSON.parse(stored) : [];
    } catch { return []; }
  });
  useEffect(() => {
    try { localStorage.setItem('textcluster_named_labels', JSON.stringify(namedLabels)); } catch {}
  }, [namedLabels]);
  const handleRemoveNamedLabel = useCallback((id) => {
    setNamedLabels((prev) => prev.filter((l) => l.id !== id));
  }, []);

  const [selections, setSelections] = useState([]);
  const [activeSelectionId, setActiveSelectionId] = useState(null);
  const [currentSelection, setCurrentSelection] = useState(null);

  const [explanation, setExplanation] = useState('');
  const [clusterLabel, setClusterLabel] = useState('');
  const [selectionColor, setSelectionColor] = useState(null);
  const [isExplaining, setIsExplaining] = useState(false);
  const [explainError, setExplainError] = useState(null);
  const [explainSettings, setExplainSettings] = useState({ model: 'gpt-5-nano-2025-08-07', promptTemplate: '' });

  const [phrases, setPhrases] = useState([]);
  const [isPhraseClouding, setIsPhraseClouding] = useState(false);
  const [phrasecloudError, setPhrasecloudError] = useState(null);
  const [phraseHighlight, setPhraseHighlight] = useState(null); // { phrase, indices }

  const [testPoints, setTestPoints] = useState([]);
  const [isReprojecting, setIsReprojecting] = useState(false);
  const [reprojectError, setReprojectError] = useState(null);

  const [hoveredEntryIndex, setHoveredEntryIndex] = useState(null);
  const [clickedEntryIndex, setClickedEntryIndex] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [controlsCollapsed, setControlsCollapsed] = useState(false);

  const [dotSize, setDotSize] = useState(4);
  const [opacity, setOpacity] = useState(0.7);
  const [showLabels, setShowLabels] = useState(true);
  const [labelFontSize, setLabelFontSize] = useState(11);
  const [whiteBg, setWhiteBg] = useState(false);
  const [colorByClusters, setColorByClusters] = useState(true);
  const [lassoMode, setLassoMode] = useState(true);
  const [highlightedCluster, setHighlightedCluster] = useState(null);
  const resetZoomRef = useRef(null);
  const [leftWidth, setLeftWidth] = useState(248);
  const [rightWidth, setRightWidth] = useState(400);
  const [tooltip, setTooltip] = useState({ visible: false, x: 0, y: 0, text: '' });

  useEffect(() => {
    setCurrentSelection(null);
    setExplanation('');
    setClusterLabel('');
    setSelectionColor(null);
    setExplainError(null);
    setActiveSelectionId(null);
    setTestPoints([]);
    setHighlightedCluster(null);
    setPhrases([]);
    setPhrasecloudError(null);
    setPhraseHighlight(null);
    setSearchQuery('');
  }, [currentDataset]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') setPhraseHighlight(null); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // Pre-load LM data in background whenever dataset changes (so toggle is instant)
  useEffect(() => {
    lm.handleDatasetChange(currentDataset);
    setLmPinnedIndex(null);
    setLmClickedPointIndex(null);
    setLmCurrentSelection(null);
    setLmExplanation('');
    setLmExplainError(null);
    setLmClusterLabel('');
    setLmSelectionColor(null);
  }, [currentDataset]); // eslint-disable-line

  const handleSelection = useCallback(
    async (indices) => {
      if (!datasetData || indices.length === 0) return;
      const texts = indices.map((i) => datasetData.texts[i]);
      setCurrentSelection({ indices, texts });
      setExplanation('');
      setClusterLabel('');
      setSelectionColor(null);
      setExplainError(null);
      setActiveSelectionId(null);
      setPhraseHighlight(null);
      setClickedEntryIndex(null);
      setHoveredEntryIndex(null);

      if (indices.length >= 3) {
        setIsPhraseClouding(true);
        setPhrasecloudError(null);
        setPhrases([]);
        try {
          const res = await axios.post('/api/phrasecloud', {
            dataset: currentDataset,
            selected_indices: indices,
          });
          if (res.data.error) {
            setPhrasecloudError(res.data.error);
          } else {
            setPhrases(res.data.phrases || []);
          }
        } catch (err) {
          setPhrasecloudError(
            err.response?.data?.error || err.message || 'Phrasecloud unavailable'
          );
        } finally {
          setIsPhraseClouding(false);
        }
      } else {
        setPhrases([]);
        setPhrasecloudError(null);
      }
    },
    [datasetData, currentDataset]
  );

  const handleExplain = useCallback(async () => {
    if (!currentSelection || !datasetData) return;
    setIsExplaining(true);
    setExplainError(null);
    const { indices, texts } = currentSelection;
    const selectedSet = new Set(indices);
    const otherTexts = datasetData.texts.filter((_, i) => !selectedSet.has(i));
    const shuffled = otherTexts.sort(() => Math.random() - 0.5).slice(0, 40);
    try {
      const explainBody = {
        dataset: currentDataset,
        selected_texts: texts,
        all_texts_sample: shuffled,
        model: explainSettings.model,
      };
      if (explainSettings.promptTemplate.trim()) {
        explainBody.prompt_template = explainSettings.promptTemplate;
      }
      const res = await axios.post('/api/explain', explainBody);
      const explanationText = res.data.explanation || '';
      const labelText = res.data.cluster_label || '';
      const color = randomSelectionColor();
      setExplanation(explanationText);
      setSelectionColor(color);
      const newEntry = {
        id: Date.now().toString(),
        dataset: currentDataset,
        indices,
        texts,
        explanation: explanationText,
        clusterLabel: labelText,
        selectionColor: color,
        phrases,
        timestamp: Date.now(),
      };
      setSelections((prev) => [newEntry, ...prev]);
      setActiveSelectionId(newEntry.id);
      if (labelText) {
        setNamedLabels((prev) => [...prev, {
          id: newEntry.id,
          dataset: currentDataset,
          indices,
          label: labelText,
          color,
        }]);
      }
    } catch (err) {
      setExplainError(err.response?.data?.error || err.message || 'Failed to get explanation');
    } finally {
      setIsExplaining(false);
    }
  }, [currentSelection, datasetData, currentDataset, phrases, explainSettings]);

  const handleCancelSelection = useCallback(() => {
    setCurrentSelection(null);
    setExplanation('');
    setClusterLabel('');
    setSelectionColor(null);
    setExplainError(null);
    setActiveSelectionId(null);
    setPhrases([]);
    setPhrasecloudError(null);
    setPhraseHighlight(null);
    setClickedEntryIndex(null);
    setHoveredEntryIndex(null);
  }, []);

  const handleRestoreSelection = useCallback(
    (sel) => {
      setCurrentSelection({ indices: sel.indices, texts: sel.texts });
      setExplanation(sel.explanation);
      setClusterLabel(sel.clusterLabel || '');
      setSelectionColor(sel.selectionColor || null);
      setPhrases(sel.phrases || []);
      setPhrasecloudError(null);
      setExplainError(null);
      setPhraseHighlight(null);
      setActiveSelectionId(sel.id);
      if (sel.dataset !== currentDataset) setCurrentDataset(sel.dataset);
    },
    [currentDataset]
  );

  const handleDeleteSelection = useCallback(
    (id) => {
      setSelections((prev) => prev.filter((s) => s.id !== id));
      setNamedLabels((prev) => prev.filter((l) => l.id !== id));
      if (activeSelectionId === id) {
        setCurrentSelection(null);
        setExplanation('');
        setExplainError(null);
        setActiveSelectionId(null);
      }
    },
    [activeSelectionId]
  );

  const handleEntryClick = useCallback((idx) => {
    setClickedEntryIndex((prev) => (prev === idx ? null : idx));
  }, []);

  const handleExportCsv = useCallback(() => {
    if (!currentSelection || !datasetData) return;
    const { indices, texts } = currentSelection;
    const semLabels = datasetData.semantic_labels || [];
    const rows = [['index', 'text', 'cluster_id', 'cluster_label']];
    indices.forEach((idx, i) => {
      const clusterId = datasetData.labels[idx] ?? '';
      const clusterLabel = clusterId !== '' && clusterId !== -1 ? (semLabels[clusterId] ?? clusterId) : '';
      const escaped = texts[i].replace(/"/g, '""');
      rows.push([idx, `"${escaped}"`, clusterId, `"${clusterLabel}"`]);
    });
    const csv = rows.map((r) => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `selection-${currentDataset}-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [currentSelection, datasetData, currentDataset]);

  const handlePhraseClick = useCallback(
    (phrase) => {
      if (!datasetData) return;
      if (phraseHighlight?.phrase === phrase) {
        setPhraseHighlight(null);
        return;
      }
      const lower = phrase.toLowerCase();
      const indices = datasetData.texts.reduce((acc, text, i) => {
        if (text.toLowerCase().includes(lower)) acc.push(i);
        return acc;
      }, []);
      setPhraseHighlight({ phrase, indices });
    },
    [datasetData, phraseHighlight]
  );

  const handleSaveState = useCallback(() => {
    const state = { version: 1, currentDataset, selections, testPoints };
    const blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `textcluster-state-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [currentDataset, selections, testPoints]);

  const handleLoadState = useCallback((file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const state = JSON.parse(e.target.result);
        if (state.currentDataset) setCurrentDataset(state.currentDataset);
        if (Array.isArray(state.selections)) setSelections(state.selections);
        if (Array.isArray(state.testPoints)) setTestPoints(state.testPoints);
      } catch (err) {
        console.error('Failed to load state:', err);
      }
    };
    reader.readAsText(file);
  }, []);

  const handleReproject = useCallback(
    async (text) => {
      setIsReprojecting(true);
      setReprojectError(null);
      try {
        const res = await axios.post('/api/reproject', { dataset: currentDataset, text });
        setTestPoints((prev) => [...prev, { text, coord: res.data.coord }]);
      } catch (err) {
        setReprojectError(err.response?.data?.error || err.message || 'Failed to reproject');
      } finally {
        setIsReprojecting(false);
      }
    },
    [currentDataset]
  );

  const startLeftResize = useCallback((e) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = leftWidth;
    const onMove = (e) => setLeftWidth(Math.max(200, Math.min(600, startW + e.clientX - startX)));
    const onUp = () => { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [leftWidth]);

  const startRightResize = useCallback((e) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = rightWidth;
    const onMove = (e) => setRightWidth(Math.max(250, Math.min(600, startW - (e.clientX - startX))));
    const onUp = () => { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [rightWidth]);

  const searchMatchIndices = useMemo(() => {
    if (!searchQuery.trim() || !datasetData) return null;
    const lower = searchQuery.toLowerCase();
    return datasetData.texts.reduce((acc, text, i) => {
      if (text.toLowerCase().includes(lower)) acc.push(i);
      return acc;
    }, []);
  }, [searchQuery, datasetData]);

  return (
    <div className="app">
      {/* ── Toolbar ── */}
      <header className="app__toolbar">
        <button
          className="app__title-toggle"
          onClick={() => setAppMode(appMode === 'explain' ? 'latent' : 'explain')}
          title="Switch mode"
        >
          <span className="app__title-text">
            {appMode === 'explain' ? 'TextCluster Explainer' : 'Latent Manipulator'}
          </span>
        </button>

        {appMode === 'explain' && (
          <>
            <div className="toolbar__divider" />
            <DatasetSelector currentDataset={currentDataset} onChange={setCurrentDataset} />
            <button onClick={handleSaveState} className="btn btn--ghost btn--sm" title="Save selections and test points to a JSON file">↓ Save</button>
            <label className="btn btn--ghost btn--sm" style={{ cursor: 'pointer' }} title="Load a previously saved session from a JSON file">
              ↑ Load
              <input
                type="file" accept=".json" style={{ display: 'none' }}
                onChange={(e) => e.target.files[0] && handleLoadState(e.target.files[0])}
              />
            </label>
            <div className="toolbar__spacer" />
            <DisplaySettings
              dotSize={dotSize} setDotSize={setDotSize}
              opacity={opacity} setOpacity={setOpacity}
              showLabels={showLabels} setShowLabels={setShowLabels}
              labelFontSize={labelFontSize} setLabelFontSize={setLabelFontSize}
              whiteBg={whiteBg} setWhiteBg={setWhiteBg}
              colorByClusters={colorByClusters} setColorByClusters={setColorByClusters}
            />
          </>
        )}
        {appMode === 'latent' && (
          <>
            <div className="toolbar__divider" />
            <DatasetSelector currentDataset={currentDataset} onChange={setCurrentDataset} />
            <button onClick={handleSaveState} className="btn btn--ghost btn--sm" title="Save selections and test points to a JSON file">↓ Save</button>
            <label className="btn btn--ghost btn--sm" style={{ cursor: 'pointer' }} title="Load a previously saved session from a JSON file">
              ↑ Load
              <input
                type="file" accept=".json" style={{ display: 'none' }}
                onChange={(e) => e.target.files[0] && handleLoadState(e.target.files[0])}
              />
            </label>
            <div className="toolbar__spacer" />
            <LMDisplaySettings
              colorMode={lm.colorMode}
              onColorModeChange={lm.setColorMode}
              showLabels={lm.showLabels}
              onShowLabelsChange={lm.setShowLabels}
              dotSize={lm.dotSize}
              onDotSizeChange={lm.setDotSize}
              opacity={lmOpacity}
              onOpacityChange={setLmOpacity}
              labelFontSize={lmLabelFontSize}
              onLabelFontSizeChange={setLmLabelFontSize}
              whiteBg={lmWhiteBg}
              onWhiteBgChange={setLmWhiteBg}
              showTrail={lm.showTrail}
              onToggleTrail={lm.toggleTrail}
              hasPinned={lmPinnedIndex !== null}
              hasCosine={lm.cosineSims.length > 0}
              hasMovement={lm.movementScores.length > 0}
              disabled={lm.disabled}
            />
          </>
        )}
      </header>

      {/* ── Three-column body (always present) ── */}
      <div className="app__body">

        {/* Left panel */}
        <aside className="app__left-panel" style={{ width: leftWidth }}>
          {appMode === 'explain' ? (
            <>
              <SelectionsHistoryPanel
                selections={selections}
                activeSelectionId={activeSelectionId}
                onRestore={handleRestoreSelection}
                onDelete={handleDeleteSelection}
              />
              {currentSelection?.texts.length > 0 && (
                <SelectedTextsAccordion
                  texts={currentSelection.texts}
                  indices={currentSelection.indices}
                  onHover={setHoveredEntryIndex}
                  onHoverOut={() => setHoveredEntryIndex(null)}
                  onClickEntry={handleEntryClick}
                  clickedEntryIndex={clickedEntryIndex}
                  onExportCsv={handleExportCsv}
                />
              )}
            </>
          ) : (
            <>
              <LMLeftPanel
                concepts={lm.concepts}
                selectedDataset={lm.selectedDataset}
                selectedConcept={lm.selectedConcept}
                onConceptChange={lm.handleConceptChange}
                onAddConcept={() => setShowLMModal(true)}
                factor={lm.factor}
                minFactor={lm.MIN_FACTOR}
                maxFactor={lm.MAX_FACTOR}
                onFactorChange={lm.handleFactorChange}
                disabled={lm.disabled}
              />
              {lmCurrentSelection?.texts.length > 0 && (
                <SelectedTextsAccordion
                  texts={lmCurrentSelection.texts}
                  indices={lmCurrentSelection.indices}
                  onHover={() => {}}
                  onHoverOut={() => {}}
                  onClickEntry={() => {}}
                  clickedEntryIndex={null}
                  onExportCsv={null}
                />
              )}
            </>
          )}
        </aside>
        <div className="resize-handle resize-handle--left" onMouseDown={startLeftResize} />

        {/* Center: Plot (always same position) */}
        <div className={`app__plot-area${(appMode === 'explain' ? whiteBg : lmWhiteBg) ? ' app__plot-area--white' : ''}`}>
          {appMode === 'explain' ? (
            <>
              {dataLoading && (
                <div className="app__loading">
                  <span className="spinner spinner--large" />
                  <p>Loading dataset…</p>
                </div>
              )}
              {dataError && (
                <div className="app__error">
                  <p>Error loading dataset: {dataError}</p>
                  <p className="app__error-hint">Make sure the backend is running on port 5001.</p>
                </div>
              )}
              {!dataLoading && !dataError && datasetData && (
                <ScatterPlot
                  data={datasetData}
                  selectedPoints={currentSelection?.indices || []}
                  testPoints={testPoints}
                  dotSize={dotSize}
                  opacity={opacity}
                  showLabels={showLabels}
                  labelFontSize={labelFontSize}
                  lassoMode={lassoMode}
                  highlightedCluster={highlightedCluster}
                  onSelection={handleSelection}
                  onTooltip={setTooltip}
                  resetZoomRef={resetZoomRef}
                  phraseHighlightIndices={phraseHighlight?.indices || null}
                  hoveredEntryIndex={hoveredEntryIndex}
                  clickedEntryIndex={clickedEntryIndex}
                  searchMatchIndices={searchMatchIndices}
                  colorByClusters={colorByClusters}
                  selectionColor={selectionColor}
                  onPointClick={(idx) => setClickedEntryIndex((prev) => (prev === idx ? null : idx))}
                  namedLabels={namedLabels.filter((l) => l.dataset === currentDataset)}
                  onRemoveLabel={handleRemoveNamedLabel}
                />
              )}
              {!dataLoading && !dataError && datasetData && (
                <div className="search-panel">
                  <span className="search-panel__icon">⌕</span>
                  <input
                    className="search-panel__input"
                    type="text"
                    placeholder="Search texts…"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    title="Highlight all points whose text contains this string"
                  />
                  {searchQuery && (
                    <button className="search-panel__clear" onClick={() => setSearchQuery('')} title="Clear search">✕</button>
                  )}
                  {searchMatchIndices && (
                    <span className="search-panel__count">{searchMatchIndices.length}</span>
                  )}
                </div>
              )}
              {!dataLoading && !dataError && datasetData && (
                <div className="plot-controls-panel">
                  <button
                    className="plot-controls-panel__minimize"
                    onClick={() => setControlsCollapsed((v) => !v)}
                    title={controlsCollapsed ? 'Expand controls' : 'Minimize controls'}
                  >
                    {controlsCollapsed ? '▾' : '▴'}
                  </button>
                  {!controlsCollapsed && (
                    <div className="plot-controls-panel__btns">
                      <button
                        className={`plot-controls-panel__btn ${lassoMode ? 'plot-controls-panel__btn--active' : ''}`}
                        onClick={() => setLassoMode((v) => !v)}
                        title={lassoMode ? 'Switch to pan/zoom mode' : 'Switch to lasso selection mode'}
                      >
                        {lassoMode ? '⬡ Lasso' : '✥ Pan/Zoom'}
                      </button>
                      <button
                        className="plot-controls-panel__btn btn--ghost-sm"
                        onClick={() => resetZoomRef.current?.()}
                        title="Reset pan and zoom to the default view"
                      >
                        ↺ Reset Zoom
                      </button>
                      {currentSelection && (
                        <button
                          className="plot-controls-panel__btn plot-controls-panel__btn--danger"
                          onClick={handleCancelSelection}
                          title="Clear the current lasso selection"
                        >
                          ✕ Selection
                        </button>
                      )}
                    </div>
                  )}
                </div>
              )}
              {!dataLoading && !dataError && datasetData?.semantic_labels?.length > 0 && (
                <div className="legend-float-panel">
                  <Legend
                    semanticLabels={datasetData.semantic_labels}
                    highlightedCluster={highlightedCluster}
                    onHighlight={setHighlightedCluster}
                  />
                </div>
              )}
            </>
          ) : (
            /* Latent Manipulator — reuse same D3 ScatterPlot */
            <>
              <ScatterPlot
                data={lm.lmBaseScatterData || datasetData}
                liveCoords={lm.hasData ? lm.currentCoords : undefined}
                liveLabels={lm.hasData ? lm.currentLabels : undefined}
                selectedPoints={lmCurrentSelection?.indices || []}
                testPoints={null}
                dotSize={lm.dotSize}
                opacity={lmOpacity}
                showLabels={lm.showLabels}
                labelFontSize={lmLabelFontSize}
                lassoMode={lmLassoMode}
                highlightedCluster={lmHighlightedCluster}
                onSelection={handleLMSelection}
                onTooltip={setTooltip}
                resetZoomRef={resetZoomRef}
                phraseHighlightIndices={null}
                hoveredEntryIndex={null}
                clickedEntryIndex={null}
                searchMatchIndices={null}
                colorByClusters={lm.colorMode === 'cluster'}
                selectionColor={lmSelectionColor}
                onPointClick={(idx) => {
                  setLmClickedPointIndex(idx);
                  if (!lmLassoMode) setLmPinnedIndex(idx);
                }}
                namedLabels={namedLabels.filter((l) => l.dataset === currentDataset)}
                onRemoveLabel={handleRemoveNamedLabel}
                pointSimilarities={lm.colorMode === 'similarity' ? lm.cosineSims : null}
                colorBySimilarity={lm.colorMode === 'similarity'}
                pointMovements={lm.colorMode === 'movement' ? lm.movementScores : null}
                colorByMovement={lm.colorMode === 'movement'}
                pinnedTrail={lmPinnedTrail}
                showColorBar={true}
              />
              {lm.hasData && lm.colorMode === 'cluster' && lmSemanticLabels.length > 0 && (
                <div className="legend-float-panel">
                  <Legend
                    semanticLabels={lmSemanticLabels}
                    highlightedCluster={lmHighlightedCluster}
                    onHighlight={setLmHighlightedCluster}
                  />
                </div>
              )}
              {lm.loading && (
                <div className="lm-loading-badge">
                  <span className="spinner spinner--sm" /> {lm.loadingMsg}
                </div>
              )}
              <div className="plot-controls-panel">
                <button
                  className="plot-controls-panel__minimize"
                  onClick={() => setControlsCollapsed((v) => !v)}
                  title={controlsCollapsed ? 'Expand controls' : 'Minimize controls'}
                >
                  {controlsCollapsed ? '▾' : '▴'}
                </button>
                {!controlsCollapsed && (
                  <div className="plot-controls-panel__btns">
                    <button
                      className={`plot-controls-panel__btn ${lmLassoMode ? 'plot-controls-panel__btn--active' : ''}`}
                      onClick={() => setLmLassoMode((v) => !v)}
                      title={lmLassoMode ? 'Switch to pan/zoom mode' : 'Switch to lasso selection mode'}
                    >
                      {lmLassoMode ? '⬡ Lasso' : '✥ Pan/Zoom'}
                    </button>
                    <button
                      className="plot-controls-panel__btn btn--ghost-sm"
                      onClick={() => resetZoomRef.current?.()}
                      title="Reset pan and zoom"
                    >
                      ↺ Reset Zoom
                    </button>
                    {lmCurrentSelection && (
                      <button
                        className="plot-controls-panel__btn plot-controls-panel__btn--danger"
                        onClick={handleLMCancelSelection}
                        title="Clear the current lasso selection"
                      >
                        ✕ Selection
                      </button>
                    )}
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        <div className="resize-handle resize-handle--right" onMouseDown={startRightResize} />

        {/* Right panel */}
        <aside className="app__right-panel" style={{ width: rightWidth, ...(appMode === 'latent' ? { padding: 0, gap: 0 } : {}) }}>
          {appMode === 'explain' ? (
            <>
              <div className="right-panel__card right-panel__card--phrasecloud">
                <PhraseCloud
                  phrases={phrases}
                  isLoading={isPhraseClouding}
                  error={phrasecloudError}
                  phraseHighlight={phraseHighlight}
                  onPhraseClick={handlePhraseClick}
                  staticMode={staticMode}
                />
              </div>
              <div className="right-panel__card right-panel__card--explanation">
                <ExplanationPanel
                  selectedCount={currentSelection?.texts.length || 0}
                  explanation={explanation}
                  isExplaining={isExplaining}
                  explainError={explainError}
                  canExplain={!!currentSelection && !isExplaining}
                  onExplain={handleExplain}
                  onClear={handleCancelSelection}
                  staticMode={staticMode}
                  explainSettings={explainSettings}
                  onExplainSettingsChange={setExplainSettings}
                />
              </div>
              <div className="right-panel__card right-panel__card--testinput">
                <TestInputPanel
                  testPoints={testPoints}
                  isReprojecting={isReprojecting}
                  reprojectError={reprojectError}
                  onSubmit={handleReproject}
                  onClearAll={() => setTestPoints([])}
                  staticMode={staticMode}
                />
              </div>
            </>
          ) : (
            <div className="lm-right-split">
              {/* Top half: selected point details */}
              <div className="lm-right-split__top">
                <div className="lm-panel-section-header">Selected Point</div>
                {lmClickedPointIndex !== null && lm.texts.length > 0 ? (
                  <div className="lm-point-details">
                    <div className="lm-point-details__header">
                      <div className="lm-point-details__meta">
                        <span className="lm-point-details__index">#{lmClickedPointIndex}</span>
                        {(() => {
                          const clusterId = lm.currentLabels[lmClickedPointIndex];
                          if (clusterId === undefined) return null;
                          if (clusterId === -1) return <span className="lm-point-details__badge lm-point-details__badge--noise">Noise</span>;
                          const topic = lm.currentTopics[clusterId];
                          return <span className="lm-point-details__badge">{topic || `Cluster ${clusterId}`}</span>;
                        })()}
                      </div>
                      <button className="lm-point-details__close" onClick={() => setLmClickedPointIndex(null)}>✕</button>
                    </div>
                    {lm.cosineSims[lmClickedPointIndex] != null && (
                      <div className="lm-point-details__sim">
                        Similarity: <strong>{lm.cosineSims[lmClickedPointIndex].toFixed(3)}</strong>
                      </div>
                    )}
                    <div className="lm-point-details__text">
                      {lm.texts[lmClickedPointIndex]}
                    </div>
                  </div>
                ) : (
                  <div className="lm-point-details lm-point-details--empty">
                    <span className="lm-point-details__empty-icon">⊙</span>
                    <p>Click a point to view details</p>
                  </div>
                )}
              </div>

              {/* Bottom half: explanation */}
              <div className="lm-right-split__bottom">
                <div className="right-panel__card right-panel__card--explanation">
                  <ExplanationPanel
                    selectedCount={lmCurrentSelection?.texts.length || 0}
                    explanation={lmExplanation}
                    isExplaining={lmIsExplaining}
                    explainError={lmExplainError}
                    canExplain={!!lmCurrentSelection && !lmIsExplaining}
                    onExplain={handleLMExplain}
                    onClear={handleLMCancelSelection}
                    staticMode={false}
                    explainSettings={lmExplainSettings}
                    onExplainSettingsChange={setLmExplainSettings}
                  />
                </div>
              </div>
            </div>
          )}
        </aside>
      </div>

      <Tooltip visible={tooltip.visible} x={tooltip.x} y={tooltip.y} text={tooltip.text} />

      {/* LM modal + toast */}
      {showLMModal && (
        <LMNewConceptModal
          onClose={() => setShowLMModal(false)}
          onSubmit={lm.handleCreateConcept}
        />
      )}
      {lm.activeJob && lm.activeJob.status !== 'completed' && lm.activeJob.status !== 'failed' && (
        <LMWorkerToast job={lm.activeJob} />
      )}

      <AppFooter appMode={appMode} />
    </div>
  );
}

function AppFooter({ appMode }) {
  const [helpOpen, setHelpOpen] = React.useState(false);
  const popoverRef = React.useRef(null);

  React.useEffect(() => {
    if (!helpOpen) return;
    const handler = (e) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target)) setHelpOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [helpOpen]);

  return (
    <footer className="app__footer">
      <span className="app__footer-tagline">
        {appMode === 'explain'
          ? 'Interactive text embedding explorer — cluster, compare, and explain your corpus'
          : 'Steer embedding visualizations along user-defined concept directions'}
      </span>
      <div className="app__footer-actions">
        <a
          className="app__footer-link"
          href="https://github.com/shivam-raval96/TextCluster-Explainer"
          target="_blank"
          rel="noreferrer"
          title="View source on GitHub"
        >
          GitHub ↗
        </a>
        <div className="app__footer-help-wrap" ref={popoverRef}>
          <button
            className="app__footer-help-btn"
            onClick={() => setHelpOpen((v) => !v)}
            title="How to use TextCluster Explainer"
          >
            ?
          </button>
          {helpOpen && (
            <div className="app__footer-popover">
              <p className="app__footer-popover-title">How to use</p>
              <ol className="app__footer-popover-steps">
                <li><strong>Pick a dataset</strong> from the toolbar dropdown.</li>
                <li><strong>Lasso points</strong> on the map to select a cluster of texts.</li>
                <li><strong>Get Explanation</strong> — the AI describes what those texts share.</li>
                <li><strong>Phrase cloud</strong> shows the most distinctive n-grams in your selection vs. the rest of the corpus (requires ≥ 3 points).</li>
                <li><strong>Click a phrase</strong> to highlight all points containing it.</li>
                <li><strong>Test new input</strong> — type any text to see where it lands in the embedding space.</li>
                <li><strong>Clusters panel</strong> (bottom-left) — click a cluster name to isolate it on the map.</li>
                <li><strong>History</strong> (left panel) — restore or delete past selections.</li>
                <li><strong>↓ CSV</strong> — export the selected texts with cluster labels.</li>
              </ol>
              <p className="app__footer-popover-note">
                Built with SentenceBERT · UMAP · LinearSVC · GPT-5 nano
              </p>
            </div>
          )}
        </div>
      </div>
    </footer>
  );
}

export default App;
