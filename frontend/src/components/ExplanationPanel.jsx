import React, { useState } from 'react';

export function SelectedTextsAccordion({ texts, indices, onHover, onHoverOut, onClickEntry, clickedEntryIndex }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="sel-panel__accordion">
      <button
        className="sel-panel__accordion-toggle"
        onClick={() => setOpen((v) => !v)}
      >
        <span>{open ? '▾' : '▸'} Selected texts</span>
        <span className="sel-panel__accordion-count">{texts.length}</span>
      </button>
      {open && (
        <ul className="sel-panel__texts-list">
          {texts.map((t, i) => {
            const idx = indices?.[i];
            const isActive = idx !== undefined && clickedEntryIndex === idx;
            return (
              <li
                key={i}
                className={`sel-panel__texts-item${isActive ? ' sel-panel__texts-item--active' : ''}`}
                onMouseEnter={() => idx !== undefined && onHover?.(idx)}
                onMouseLeave={() => onHoverOut?.()}
                onClick={() => idx !== undefined && onClickEntry?.(idx)}
              >
                {t.length > 90 ? t.slice(0, 87) + '…' : t}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function ExplanationPanel({
  selectedCount,
  explanation,
  isExplaining,
  explainError,
  canExplain,
  onExplain,
  onClear,
  staticMode,
}) {
  return (
    <div className="sel-panel">
      <div className="sel-panel__header">
        <span className="sel-panel__title">Natural language explanation</span>
        {selectedCount > 0 && (
          <span className="sel-panel__badge">{selectedCount}</span>
        )}
        {selectedCount > 0 && (
          <button className="btn btn--secondary btn--sm" onClick={onClear} title="Clear the current lasso selection" style={{ marginLeft: 'auto' }}>
            ✕
          </button>
        )}
      </div>

      <div className="sel-panel__body">
        {staticMode ? (
          <p className="sel-panel__backend-notice">
            <span className="sel-panel__backend-icon">⚡</span>
            Backend not connected — AI explanations require the live server.
          </p>
        ) : (
          <>
            <div className="sel-panel__explain-row">
              <button
                className="btn btn--primary btn--sm"
                onClick={onExplain}
                disabled={!canExplain}
                title="Send the selected texts to the AI model for a natural language explanation"
              >
                {isExplaining ? (
                  <><span className="spinner spinner--small" /> Generating…</>
                ) : (
                  'Get Explanation'
                )}
              </button>
            </div>
            {!selectedCount && !explanation && !isExplaining && !explainError && (
              <p className="sel-panel__empty">
                Select points with the lasso tool, then click <strong>Get Explanation</strong>.
              </p>
            )}

            {isExplaining && (
              <div className="sel-panel__loading">
                <span className="spinner spinner--small" />
                <span>Generating explanation…</span>
              </div>
            )}

            {!isExplaining && explainError && (
              <p className="sel-panel__error">{explainError}</p>
            )}

            {!isExplaining && explanation && (
              <div className="sel-panel__explanation">
                <div className="sel-panel__explanation-header">✦ AI Explanation</div>
                {explanation}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default ExplanationPanel;
