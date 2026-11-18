import { useEffect, useState } from 'react'
import { api } from '../api'

const f3 = (x) => (x == null ? '—' : x.toFixed(3))

export default function ModelCard() {
  const [m, setM] = useState(null)
  useEffect(() => { api.modelCard().then(setM) }, [])
  if (!m) return <div className="page">Loading…</div>
  const c = m.classification, s = m.segmentation, d = m.data_summary
  return (
    <div className="page card">
      <h2>Model card & limitations</h2>
      <p className="warn">{m.disclaimer}</p>
      {d && (
        <p className="muted">
          Chest X-ray: {d.images.toLocaleString()} images / {d.patients.toLocaleString()} patients, split by patient ·
          {' '}CT: {Object.values(d.ct_volumes).reduce((a, b) => a + b, 0)} spleen-labelled volumes
        </p>
      )}
      {c && (
        <>
          <h3>Chest X-ray classifier — test set ({c.test_images} images), macro-AUROC {f3(c.macro_auroc)}</h3>
          <table className="audit">
            <thead><tr><th>Finding</th><th>AUROC</th><th>Sensitivity</th><th>Specificity</th><th>Positives</th></tr></thead>
            <tbody>
              {Object.entries(c.per_finding).map(([k, v]) => (
                <tr key={k}><td>{k}</td><td>{f3(v.auroc)}</td><td>{f3(v.sensitivity)}</td><td>{f3(v.specificity)}</td><td>{v.positives}</td></tr>
              ))}
            </tbody>
          </table>
          <h3>Grad-CAM localisation vs radiologist boxes</h3>
          <p>Pointing game {f3(c.gradcam_localisation.pointing_game)} over {c.gradcam_localisation.boxes} boxes · mean IoU {f3(c.gradcam_localisation.mean_iou)}</p>
          <h3>Subgroup performance (bias audit)</h3>
          <table className="audit">
            <thead><tr><th>Group</th><th>n</th><th>macro-AUROC</th></tr></thead>
            <tbody>
              {Object.entries(c.subgroups).map(([k, v]) => <tr key={k}><td>{k}</td><td>{v.n}</td><td>{f3(v.macro_auroc)}</td></tr>)}
            </tbody>
          </table>
          <p>Cardiomegaly false-positive rate by view: {Object.entries(c.cardiomegaly_false_positive_rate_by_view).map(([k, v]) => `${k} ${(v * 100).toFixed(1)}%`).join(' · ')}
            {' '}— AP (portable) films magnify the heart, a known confounder.</p>
        </>
      )}
      {s && <h3>Spleen segmentation — {s.test_volumes} test CT volumes: mean Dice {f3(s.mean_dice)}, HD95 {s.mean_hd95_mm.toFixed(1)} mm</h3>}
      <h3>Known limitations</h3>
      <ul>
        <li>Not validated for any clinical purpose; outputs must never be used for diagnosis or treatment.</li>
        <li>NIH ChestX-ray14 labels were mined from reports by NLP and are noisy; the model inherits that noise.</li>
        <li>Single-site, single-country source population; performance on other hospitals, devices or paediatric patients is unknown.</li>
        <li>Only 8 findings are modelled; absence of an alert does not mean a normal study.</li>
        <li>Grad-CAM shows where the model looked, not why; coarse maps can highlight correlated regions (e.g. devices, text markers).</li>
        <li>Spleen segmentation was trained on portal-venous abdominal CT only.</li>
      </ul>
    </div>
  )
}
