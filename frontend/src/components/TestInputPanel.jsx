import React, { useState } from 'react';

function TestInputPanel({ testPoints, isReprojecting, reprojectError, onSubmit, onClearAll, staticMode }) {
  const [inputText, setInputText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = inputText.trim();
    if (!trimmed || isReprojecting) return;
    onSubmit(trimmed);
    setInputText('');
  };

  return (
    <div className="test-input-panel">
      {/* Header */}
      <div className="sel-panel__header">
        <span className="sel-panel__title">Test new input</span>
      </div>

      {staticMode ? (
        <p className="sel-panel__backend-notice" style={{ padding: '8px 14px 12px' }}>
          <span className="sel-panel__backend-icon">⚡</span>
          Backend not connected — re-projection requires the live server.
        </p>
      ) : (
        <>
          {/* Input form */}
          <div className="test-input-panel__form-area">
            <form className="test-input-panel__form" onSubmit={handleSubmit}>
              <input
                type="text"
                className="test-input-panel__input"
                placeholder="Type a sentence to project onto the map…"
                title="Type any text to see where it would land in the embedding space"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                disabled={isReprojecting}
              />
              <button
                type="submit"
                className="test-input-panel__btn"
                disabled={isReprojecting || !inputText.trim()}
                title="Embed this text and plot it as a star on the map"
              >
                {isReprojecting ? (
                  <><span className="spinner spinner--small" /> Projecting…</>
                ) : (
                  'Show'
                )}
              </button>
            </form>

            {reprojectError && (
              <p className="test-input-panel__error">{reprojectError}</p>
            )}
          </div>

          {/* Results */}
          <div className="test-input-panel__results">
            {testPoints.length === 0 && !reprojectError && (
              <p className="sel-panel__empty" style={{ padding: '10px 14px' }}>
                Enter text above to see where it falls in the embedding space.
              </p>
            )}
            {testPoints.length > 0 && (
              <>
                <div style={{ display: 'flex', justifyContent: 'flex-end', padding: '4px 14px 0' }}>
                  <button className="btn btn--secondary btn--sm" onClick={onClearAll} title="Remove all test points from the scatter plot">
                    Clear all
                  </button>
                </div>
                <ul className="test-input-panel__list">
                  {testPoints.map((pt, i) => (
                    <li key={i} className="test-input-panel__list-item">
                      <span className="test-input-panel__list-dot">◆</span>
                      <span className="test-input-panel__list-text">
                        {pt.text.length > 70 ? pt.text.slice(0, 67) + '…' : pt.text}
                      </span>
                      <span className="test-input-panel__list-coord">
                        [{pt.coord[0].toFixed(1)}, {pt.coord[1].toFixed(1)}]
                      </span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default TestInputPanel;
