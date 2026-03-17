import React, { useEffect, useState, useRef } from 'react';
import cloud from 'd3-cloud';

const CLOUD_H = 260;

function PhraseCloud({ phrases, isLoading, error, phraseHighlight, onPhraseClick, staticMode }) {
  const containerRef = useRef(null);
  const [cloudWords, setCloudWords] = useState([]);
  const [cloudW, setCloudW] = useState(300);

  // Track container width
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        if (w > 0) setCloudW(w);
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!phrases || phrases.length === 0) {
      setCloudWords([]);
      return;
    }

    const weights = phrases.map((p) => p.weight);
    const minW = Math.min(...weights);
    const maxW = Math.max(...weights);
    const range = maxW - minW || 1;

    const sizeScale = (w) => 8 + ((w - minW) / range) * (22 - 8);

    const words = phrases.map((p) => ({
      text: p.text,
      size: sizeScale(p.weight),
      score: p.weight,
    }));

    cloud()
      .size([cloudW, CLOUD_H])
      .words(words)
      .padding(3)
      .rotate(0)
      .font('Inter, -apple-system, sans-serif')
      .fontSize((d) => d.size)
      .on('end', (computed) => setCloudWords(computed))
      .start();
  }, [phrases, cloudW]);

  if (staticMode) {
    return (
      <div className="phrase-cloud" ref={containerRef}>
        <div className="phrase-cloud__header">
          <span className="phrase-cloud__title">Contrastive phrase cloud</span>
        </div>
        <p className="sel-panel__backend-notice">
          <span className="sel-panel__backend-icon">⚡</span>
          Backend not connected — phrase clouds require the live server.
        </p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="phrase-cloud__loading">
        <span className="spinner spinner--small" />
        <span>Generating phrase cloud…</span>
      </div>
    );
  }

  if (error) {
    return <p className="phrase-cloud__error">{error}</p>;
  }

  const activePhrase = phraseHighlight?.phrase ?? null;
  const matchCount = phraseHighlight?.indices?.length ?? 0;

  return (
    <div className="phrase-cloud" ref={containerRef}>
      <div className="phrase-cloud__header">
        <span className="phrase-cloud__title">Contrastive phrase cloud</span>
        {activePhrase && (
          <span className="phrase-cloud__match-badge">
            "{activePhrase}" — {matchCount} match{matchCount !== 1 ? 'es' : ''}
          </span>
        )}
      </div>
      {!isLoading && !error && (!phrases || phrases.length === 0) && (
        <p className="phrase-cloud__empty">Lasso 3+ points to generate a phrase cloud.</p>
      )}
      <svg
        width={cloudW}
        height={CLOUD_H}
        style={{ display: cloudWords.length > 0 ? 'block' : 'none', overflow: 'visible' }}
      >
        <g transform={`translate(${cloudW / 2},${CLOUD_H / 2})`}>
          {cloudWords.map((w, i) => (
            <text
              key={i}
              style={{
                fontSize: `${w.size}px`,
                fontFamily: w.font,
                fontWeight: 700,
                fill: activePhrase === w.text ? 'var(--accent-bright)' : '#0f172a',
                cursor: 'pointer',
                userSelect: 'none',
                outline: activePhrase === w.text ? '1px solid var(--border-accent)' : 'none',
              }}
              textAnchor="middle"
              transform={`translate(${w.x},${w.y})rotate(${w.rotate})`}
              onClick={() => onPhraseClick(w.text)}
            >
              <title>weight: {w.score?.toFixed(3)}</title>
              {w.text}
            </text>
          ))}
        </g>
      </svg>
    </div>
  );
}

export default PhraseCloud;
