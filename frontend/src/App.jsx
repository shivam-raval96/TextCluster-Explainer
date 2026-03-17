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

import './styles/App.css';

function App() {
  const [currentDataset, setCurrentDataset] = useState('vispub');
  const { data: datasetData, loading: dataLoading, error: dataError, staticMode } = useDataset(currentDataset);

  const [selections, setSelections] = useState([]);
  const [activeSelectionId, setActiveSelectionId] = useState(null);
  const [currentSelection, setCurrentSelection] = useState(null);

  const [explanation, setExplanation] = useState('');
  const [isExplaining, setIsExplaining] = useState(false);
  const [explainError, setExplainError] = useState(null);

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

  const [dotSize, setDotSize] = useState(4);
  const [opacity, setOpacity] = useState(0.7);
  const [showLabels, setShowLabels] = useState(true);
  const [labelFontSize, setLabelFontSize] = useState(11);
  const [lassoMode, setLassoMode] = useState(true);
  const [highlightedCluster, setHighlightedCluster] = useState(null);
  const resetZoomRef = useRef(null);
  const [leftWidth, setLeftWidth] = useState(248);
  const [rightWidth, setRightWidth] = useState(400);
  const [tooltip, setTooltip] = useState({ visible: false, x: 0, y: 0, text: '' });

  useEffect(() => {
    setCurrentSelection(null);
    setExplanation('');
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

  const handleSelection = useCallback(
    async (indices) => {
      if (!datasetData || indices.length === 0) return;
      const texts = indices.map((i) => datasetData.texts[i]);
      setCurrentSelection({ indices, texts });
      setExplanation('');
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
      const res = await axios.post('/api/explain', {
        dataset: currentDataset,
        selected_texts: texts,
        all_texts_sample: shuffled,
      });
      const explanationText = res.data.explanation || '';
      setExplanation(explanationText);
      const newEntry = {
        id: Date.now().toString(),
        dataset: currentDataset,
        indices,
        texts,
        explanation: explanationText,
        phrases,
        timestamp: Date.now(),
      };
      setSelections((prev) => [newEntry, ...prev]);
      setActiveSelectionId(newEntry.id);
    } catch (err) {
      setExplainError(err.response?.data?.error || err.message || 'Failed to get explanation');
    } finally {
      setIsExplaining(false);
    }
  }, [currentSelection, datasetData, currentDataset, phrases]);

  const handleCancelSelection = useCallback(() => {
    setCurrentSelection(null);
    setExplanation('');
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
        <h1 className="app__title">TextCluster Explainer</h1>

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
        />
      </header>

      {/* ── Three-column body ── */}
      <div className="app__body">

        {/* Left: History + Selected Texts + Legend */}
        <aside className="app__left-panel" style={{ width: leftWidth }}>
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
            />
          )}
          {datasetData?.semantic_labels?.length > 0 && (
            <Legend
              semanticLabels={datasetData.semantic_labels}
              highlightedCluster={highlightedCluster}
              onHighlight={setHighlightedCluster}
            />
          )}
        </aside>
        <div className="resize-handle resize-handle--left" onMouseDown={startLeftResize} />

        {/* Center: Plot */}
        <div className="app__plot-area">
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

        <div className="resize-handle resize-handle--right" onMouseDown={startRightResize} />

        {/* Right: Explain panels */}
        <aside className="app__right-panel" style={{ width: rightWidth }}>
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
        </aside>
      </div>

      <Tooltip visible={tooltip.visible} x={tooltip.x} y={tooltip.y} text={tooltip.text} />
    </div>
  );
}

export default App;
