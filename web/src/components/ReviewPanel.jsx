import { useState } from 'react'
import { api, pct, when } from '../api'

export default function ReviewPanel({ study, finding, onFinding, onDecided }) {
  const ai = study.ai || {}
  const [reviewer, setReviewer] = useState(() => localStorage.getItem('reviewer') || '')
  const [comment, setComment] = useState('')
  const [disagree, setDisagree] = useState([])
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)

  const decide = async (decision) => {
    if (!reviewer.trim()) { setMsg({ err: 'Enter your name to sign the decision.' }); return }
    setBusy(true)
    try {
      localStorage.setItem('reviewer', reviewer)
      const r = await api.decide(study.study_id, { decision, reviewer, comment: comment || null, disagree_findings: disagree })
      setMsg({ ok: `${decision === 'approved' ? 'Approved' : 'Rejected'} · audit ${r.audit_hash.slice(0, 12)}…` })
      await onDecided()
    } catch (e) {
      setMsg({ err: String(e) })
    } finally { setBusy(false) }
  }

  const toggle = (f) => setDisagree((d) => (d.includes(f) ? d.filter((x) => x !== f) : [...d, f]))

  return (
    <aside className="review-panel">
      <div className="panel-title">AI result</div>
      <div className="summary">{ai.summary}</div>
      <div className="meta">model {JSON.parse(study.model_version || '{}')[study.kind === 'ct' ? 'segmenter' : 'classifier']} · {ai.inference_seconds}s</div>

      {study.kind === 'cxr' && (
        <ul className="findings">
          {ai.findings.map((f) => (
            <li key={f.finding} className={f.finding === finding ? 'active' : ''} onClick={() => onFinding(f.finding)}>
              <div className="fname">
                {f.finding}
                {f.positive && <span className="pos">above threshold</span>}
              </div>
              <div className="pbar">
                <span className="fill" style={{ width: pct(f.probability) }} />
                <span className="thr" style={{ left: pct(f.threshold) }} title={`threshold ${pct(f.threshold)}`} />
              </div>
              <div className="pval">{pct(f.probability)}</div>
            </li>
          ))}
        </ul>
      )}
      {study.kind === 'cxr' && <div className="hint">Click a finding to see its Grad-CAM. Tick marks show each finding's decision threshold.</div>}

      {study.kind === 'ct' && (
        <dl className="ctstats">
          <dt>Spleen volume</dt><dd>{ai.spleen_volume_ml} mL</dd>
          <dt>Mean voxel confidence</dt><dd>{pct(ai.mean_confidence)}</dd>
          <dt>Slices with spleen</dt><dd>{ai.spleen_slices ? `${ai.spleen_slices[0] + 1}–${ai.spleen_slices[1] + 1}` : '—'}</dd>
        </dl>
      )}

      <div className="panel-title">Radiologist decision</div>
      {study.decision && (
        <div className={`decided ${study.decision}`}>
          {study.decision.toUpperCase()} by {study.reviewer} · {when(study.decided_at)}
          {study.decision_comment && <div className="c">“{study.decision_comment}”</div>}
        </div>
      )}
      <input className="field" placeholder="Reviewer name" value={reviewer} onChange={(e) => setReviewer(e.target.value)} />
      {study.kind === 'cxr' && ai.positives?.length > 0 && (
        <div className="disagree">
          <div className="lbl">Disagree with AI finding:</div>
          {ai.positives.map((f) => (
            <label key={f}><input type="checkbox" checked={disagree.includes(f)} onChange={() => toggle(f)} /> {f}</label>
          ))}
        </div>
      )}
      <textarea className="field" rows={3} placeholder="Comment (optional)" value={comment} onChange={(e) => setComment(e.target.value)} />
      <div className="actions">
        <button className="approve" disabled={busy} onClick={() => decide('approved')}>✓ Approve</button>
        <button className="reject" disabled={busy} onClick={() => decide('rejected')}>✕ Reject</button>
      </div>
      {msg && <div className={msg.ok ? 'ok' : 'err'}>{msg.ok || msg.err}</div>}

      <div className="panel-title">Study audit trail</div>
      <ol className="trail">
        {(study.audit || []).map((a) => (
          <li key={a.id}>
            <span className="t">{when(a.ts)}</span>
            <span className="a">{a.action.replace(/_/g, ' ')}</span>
            <span className="who">{a.actor}</span>
          </li>
        ))}
      </ol>
    </aside>
  )
}
