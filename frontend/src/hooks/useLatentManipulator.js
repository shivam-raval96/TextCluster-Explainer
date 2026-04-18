import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import axios from 'axios';
import {
  findInterpolationParams,
  interpolateCoordinates,
  getInterpolatedSnapshot,
} from '../utils/lmInterpolation';

const TRAIL_SAMPLES = 80;

export default function useLatentManipulator() {
  const [datasets, setDatasets] = useState([]);
  const [concepts, setConcepts] = useState([]);
  const [vizData, setVizData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState('Loading…');

  const [selectedDataset, setSelectedDataset] = useState('');
  const [selectedConcept, setSelectedConcept] = useState('');

  const [factor, setFactor] = useState(0);
  const [colorMode, setColorMode] = useState('cluster');
  const [showLabels, setShowLabels] = useState(true);
  const [dotSize, setDotSize] = useState(4);
  const [showTrail, setShowTrail] = useState(false);

  const [activeJob, setActiveJob] = useState(null);

  // Derived interpolated state
  const [currentCoords, setCurrentCoords] = useState([]);
  const [currentLabels, setCurrentLabels] = useState([]);
  const [currentTopics, setCurrentTopics] = useState({});
  const [currentStats, setCurrentStats] = useState(null);

  const snapshots = vizData?.snapshots ?? [];
  const cosineSims = vizData?.cosine_similarities ?? [];
  const texts = vizData?.texts ?? [];

  // Derive factor bounds from the loaded concept's projection_factors
  const minFactor = useMemo(() => {
    const pf = vizData?.projection_factors;
    return pf?.length ? Math.min(...pf) : -1;
  }, [vizData]);
  const maxFactor = useMemo(() => {
    const pf = vizData?.projection_factors;
    return pf?.length ? Math.max(...pf) : 10;
  }, [vizData]);

  // Per-point movement score: max 2D displacement from factor=0 across all snapshots
  const movementScores = useMemo(() => {
    if (!snapshots.length) return [];
    const baseSnap = snapshots.find((s) => s.factor === 0) ?? snapshots[0];
    const base = baseSnap.embeddings_2d;
    const n = base.length;
    const scores = new Float32Array(n);
    for (const snap of snapshots) {
      if (snap === baseSnap) continue;
      for (let i = 0; i < n; i++) {
        const dx = snap.embeddings_2d[i][0] - base[i][0];
        const dy = snap.embeddings_2d[i][1] - base[i][1];
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d > scores[i]) scores[i] = d;
      }
    }
    return Array.from(scores);
  }, [snapshots]);

  // Stable base data for ScatterPlot scene build (only changes when concept/dataset changes)
  const lmBaseScatterData = useMemo(() => {
    const snap = snapshots.find((s) => s.factor === 0) || snapshots[0];
    if (!snap || !texts.length) return null;
    const labels = snap.cluster_labels;
    const maxId = labels.length ? Math.max(...labels.filter((l) => l >= 0), -1) : -1;
    const semantic_labels = maxId >= 0
      ? Array.from({ length: maxId + 1 }, (_, i) => snap.cluster_topics[i] || `Cluster ${i}`)
      : [];
    return {
      coords: snap.embeddings_2d,
      labels,
      texts,
      semantic_labels,
    };
  }, [snapshots, texts]);

  // Trail for a specific point: densely sampled path across the full factor range
  const getPinnedTrail = useCallback((pinnedIndex) => {
    if (pinnedIndex === null || pinnedIndex === undefined || !snapshots.length) return null;
    const trail = [];
    for (let s = 0; s <= TRAIL_SAMPLES; s++) {
      const f = minFactor + (s / TRAIL_SAMPLES) * (maxFactor - minFactor);
      const params = findInterpolationParams(f, snapshots);
      if (!params) continue;
      const { snapshotA, snapshotB, t } = params;
      const coords = interpolateCoordinates(snapshotA, snapshotB, t);
      if (coords[pinnedIndex]) trail.push(coords[pinnedIndex]);
    }
    return trail.length >= 2 ? trail : null;
  }, [snapshots, minFactor, maxFactor]);

  // Load datasets on mount
  useEffect(() => {
    axios.get('/api/lm/datasets')
      .then(r => setDatasets(r.data.datasets || []))
      .catch(() => {});
  }, []);

  // Poll active job
  useEffect(() => {
    if (!activeJob || activeJob.status === 'completed' || activeJob.status === 'failed') return;
    const iv = setInterval(async () => {
      try {
        const r = await axios.get(`/api/lm/worker/${activeJob.job_id}/status`);
        setActiveJob(r.data);
        if (r.data.status === 'completed' && selectedDataset) {
          axios.get(`/api/lm/concepts/${selectedDataset}`).then(r2 => setConcepts(r2.data.concepts || []));
        }
      } catch {}
    }, 2000);
    return () => clearInterval(iv);
  }, [activeJob, selectedDataset]);

  // Recompute interpolated coords whenever factor or data changes
  useEffect(() => {
    if (snapshots.length === 0) return;
    const params = findInterpolationParams(factor, snapshots);
    if (!params) return;
    const { snapshotA, snapshotB, t } = params;
    setCurrentCoords(interpolateCoordinates(snapshotA, snapshotB, t));
    const snap = getInterpolatedSnapshot(snapshotA, snapshotB, t);
    setCurrentLabels(snap.cluster_labels);
    setCurrentTopics(snap.cluster_topics);
    setCurrentStats({
      total: snap.embeddings_2d.length,
      n_clusters: snap.n_clusters,
      n_noise: snap.n_noise,
      purity_metrics: snap.purity_metrics,
    });
  }, [factor, snapshots]);

  // ── Actions ──

  const handleDatasetChange = useCallback(async (ds) => {
    setSelectedDataset(ds);
    setSelectedConcept('');
    setVizData(null);
    setFactor(0);
    setConcepts([]);
    if (ds) {
      try {
        const r = await axios.get(`/api/lm/concepts/${ds}`);
        const fetched = r.data.concepts || [];
        setConcepts(fetched);
        if (fetched.length > 0) {
          const first = fetched[0].filename;
          setSelectedConcept(first);
          setLoading(true);
          setLoadingMsg('Loading visualization…');
          try {
            const r2 = await axios.get(`/api/lm/results/${ds}/${first}`);
            setVizData(r2.data);
          } catch {}
          setLoading(false);
        }
      } catch {}
    }
  }, []);

  const handleConceptChange = useCallback(async (concept) => {
    setSelectedConcept(concept);
    setFactor(0);
    if (concept && selectedDataset) {
      setLoading(true);
      setLoadingMsg('Loading visualization…');
      try {
        const r = await axios.get(`/api/lm/results/${selectedDataset}/${concept}`);
        setVizData(r.data);
      } catch {}
      setLoading(false);
    } else {
      setVizData(null);
    }
  }, [selectedDataset]);

  const handleFactorChange = useCallback((newFactor) => {
    setFactor(newFactor);
  }, []);

  const toggleTrail = useCallback(() => setShowTrail(v => !v), []);

  const handleCreateConcept = useCallback(async (conceptName, projectionFactors) => {
    if (!selectedDataset) return;
    try {
      const r = await axios.post(`/api/lm/concepts/${selectedDataset}`, {
        concept_name: conceptName,
        projection_factors: projectionFactors,
      });
      if (r.data.job_id) {
        setActiveJob({ job_id: r.data.job_id, status: 'starting', progress: 0, message: 'Initializing…', concept_name: conceptName });
      }
    } catch {}
  }, [selectedDataset]);

  return {
    // Data
    datasets, concepts, vizData, loading, loadingMsg,
    selectedDataset, selectedConcept,
    cosineSims, movementScores, texts,
    // Visualization state
    factor, colorMode, showLabels, dotSize,
    showTrail, activeJob,
    // Interpolated
    currentCoords, currentLabels, currentTopics, currentStats,
    // Derived
    snapshots,
    lmBaseScatterData,
    hasData: currentCoords.length > 0,
    disabled: snapshots.length === 0,
    MIN_FACTOR: minFactor, MAX_FACTOR: maxFactor,
    // Actions
    handleDatasetChange, handleConceptChange,
    handleFactorChange,
    setColorMode, setShowLabels, setDotSize,
    toggleTrail,
    getPinnedTrail,
    handleCreateConcept,
  };
}
