import { sliderToFactor, factorToSlider } from '../utils/lmInterpolation';

export default function LMLeftPanel({
  concepts,
  selectedDataset, selectedConcept,
  onConceptChange, onAddConcept,
  factor, minFactor, maxFactor,
  onFactorChange,
  disabled,
}) {
  const sliderVal = factorToSlider(factor, minFactor, maxFactor);

  function handleSlider(e) {
    onFactorChange(sliderToFactor(parseFloat(e.target.value), minFactor, maxFactor));
  }

  function handleFactorInput(e) {
    let v = parseFloat(e.target.value);
    if (isNaN(v)) v = 0;
    onFactorChange(Math.max(minFactor, Math.min(maxFactor, v)));
  }

  return (
    <div className="lm-left-panel">
      {/* Concept */}
      <div className="lm-panel-section">
        <div className="lm-panel-section__title">Concept Direction</div>
        <div className="lm-field">
          <select
            className="lm-field__select"
            value={selectedConcept}
            onChange={e => onConceptChange(e.target.value)}
            disabled={!selectedDataset}
          >
            <option value="">Select concept…</option>
            {concepts.map(c => (
              <option key={c.filename} value={c.filename}>
                {c.name}
              </option>
            ))}
          </select>
        </div>
        <button
          className="btn btn--ghost btn--sm lm-add-btn"
          onClick={onAddConcept}
          disabled={!selectedDataset}
        >
          + Add Concept
        </button>
      </div>

      {/* Projection factor slider */}
      <div className="lm-panel-section">
        <div className="lm-panel-section__title">
          Projection Factor
          <input
            className="lm-factor-input"
            type="text"
            value={factor.toFixed(2)}
            onChange={handleFactorInput}
            onKeyDown={e => e.key === 'Enter' && e.target.blur()}
            disabled={disabled}
          />
        </div>
        <input
          className="lm-slider"
          type="range" min="0" max="1" step="0.001"
          value={sliderVal}
          onChange={handleSlider}
          disabled={disabled}
        />
        <div className="lm-slider-labels">
          <span>−1× remove</span>
          <span>original</span>
          <span>~{maxFactor}× amplify</span>
        </div>
      </div>
    </div>
  );
}
