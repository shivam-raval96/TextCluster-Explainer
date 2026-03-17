import { useState, useEffect } from 'react';
import axios from 'axios';

const STATIC_MODE = process.env.REACT_APP_STATIC_MODE === 'true';

/**
 * useDataset
 * In static mode: fetches /data/{name}.json (bundled in public/).
 * In live mode:   fetches /api/dataset/{name} from the Flask backend.
 * Returns { data, loading, error, staticMode }.
 *
 * data shape:
 *   { texts: string[], coords: [number,number][], labels: number[], semantic_labels: string[] }
 */
function useDataset(datasetName) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!datasetName) return;

    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);

    const url = STATIC_MODE
      ? `${process.env.PUBLIC_URL}/data/${datasetName}.json`
      : `/api/dataset/${datasetName}`;

    axios
      .get(url)
      .then((res) => {
        if (!cancelled) {
          setData(res.data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          const msg =
            err.response?.data?.error ||
            err.message ||
            'Failed to load dataset';
          setError(msg);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [datasetName]);

  return { data, loading, error, staticMode: STATIC_MODE };
}

export default useDataset;
