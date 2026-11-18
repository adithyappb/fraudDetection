import { useState } from 'react'
import { pct } from '../api'

const FILTERS = [['analyzed', 'To review'], ['done', 'Reviewed'], ['all', 'All']]

export default function Worklist({ studies, selected, onSelect }) {
  const [filter, setFilter] = useState('analyzed')
  const shown = studies.filter((s) =>
    filter === 'all' ? true : filter === 'done' ? ['approved', 'rejected'].includes(s.status) : s.status === filter)
  const count = (f) => studies.filter((s) =>
    f === 'all' ? true : f === 'done' ? ['approved', 'rejected'].includes(s.status) : s.status === f).length

  return (
    <aside className="worklist">
      <div className="panel-title">Worklist</div>
      <div className="filters">
        {FILTERS.map(([k, label]) => (
          <button key={k} className={filter === k ? 'chip active' : 'chip'} onClick={() => setFilter(k)}>
            {label} <span className="count">{count(k)}</span>
          </button>
        ))}
      </div>
      <ul>
        {shown.map((s) => (
          <li key={s.study_id} className={s.study_id === selected ? 'item selected' : 'item'} onClick={() => onSelect(s.study_id)}>
            <div className="row1">
              <span className={`mod ${s.modality}`}>{s.modality}</span>
              <span className="pid">{s.patient_id}</span>
              <span className={`status ${s.status}`}>{s.status}</span>
            </div>
            <div className="row2">{s.description} · {s.patient_sex} {s.patient_age}</div>
            {s.kind === 'cxr' && s.top_finding && (
              <div className="row3">
                <span>{s.top_finding}</span>
                <span className="bar"><span style={{ width: pct(s.top_probability) }} /></span>
                <span className="num">{pct(s.top_probability)}</span>
              </div>
            )}
            {s.kind === 'ct' && <div className="row3"><span>Spleen {s.spleen_volume_ml} mL</span></div>}
          </li>
        ))}
        {!shown.length && <li className="none">Nothing here.</li>}
      </ul>
    </aside>
  )
}
