import { useState } from 'react'
import { api, pct } from '../api'

export default function Viewer({ study, finding }) {
  const isCT = study.kind === 'ct'
  const ai = study.ai || {}
  const [show, setShow] = useState(true)
  const [opacity, setOpacity] = useState(0.75)
  // the parent remounts this component per study (key), so initial state = start on the spleen
  const [slice, setSlice] = useState(() => {
    if (!isCT) return null
    const r = ai.spleen_slices
    return r ? Math.round((r[0] + r[1]) / 2) : Math.floor((ai.slices || 1) / 2)
  })

  const f = (ai.findings || []).find((x) => x.finding === finding)
  const img = api.imageUrl(study.study_id, isCT ? slice : null)
  const ov = api.overlayUrl(study.study_id, isCT ? { slice } : { finding })

  return (
    <section className="viewer">
      <div className="viewer-head">
        <div>
          <b>{study.patient_id}</b> · {study.description} · {study.modality}
        </div>
        <div className="controls">
          <label><input type="checkbox" checked={show} onChange={(e) => setShow(e.target.checked)} /> AI overlay</label>
          <label>opacity <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(+e.target.value)} /></label>
        </div>
      </div>
      <div className="stage">
        {(!isCT || slice != null) && (
          <div className="frame">
            <img className="scan" src={img} alt="scan" draggable={false} />
            {show && <img className="overlay" src={ov} alt="AI overlay" style={{ opacity }} draggable={false} />}
            <div className="corner tl">{isCT ? `Slice ${slice + 1} / ${ai.slices}` : `${study.patient_sex} ${study.patient_age}`}</div>
            <div className="corner tr">
              {isCT ? 'Spleen segmentation (3D U-Net)' : `Grad-CAM: ${finding ?? ''}${f ? ` · ${pct(f.probability)}` : ''}`}
            </div>
            <div className="corner bl">AI output — education only</div>
            {show && !isCT && <div className="legend"><span>low</span><i /><span>high influence</span></div>}
          </div>
        )}
      </div>
      {isCT && slice != null && (
        <div className="slicebar">
          <input type="range" min="0" max={ai.slices - 1} value={slice} onChange={(e) => setSlice(+e.target.value)} />
          {ai.spleen_slices && (
            <button className="chip" onClick={() => setSlice(Math.round((ai.spleen_slices[0] + ai.spleen_slices[1]) / 2))}>
              Jump to spleen ({ai.spleen_slices[0] + 1}–{ai.spleen_slices[1] + 1})
            </button>
          )}
        </div>
      )}
    </section>
  )
}
