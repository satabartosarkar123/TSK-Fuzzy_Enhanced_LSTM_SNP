#!/usr/bin/env python3
"""
Model Performance Evaluation Matrix — Actual Model Inference
=============================================================
Uses manual GradientTape training loop (avoids model.fit() hang on stateful RNNs).
1 run × 100 epochs per combination, matching notebook protocol.
"""
import os, sys, warnings, csv, time, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

import tensorflow as tf
print(f"TF {tf.__version__}", flush=True)

# ═══════════════════════════════════════════════════════════════
BASE = "/Users/satabarto/Research/content"
MF_SIGMA = 0.25; UNITS = 8; N_EPOCHS = 100; TEST_SIZE = 60

DATASETS = {
    "Dow Jones":       f"{BASE}/monthly-closings-of-the-dowjones.csv",
    "S&P 500":         f"{BASE}/sp500.csv",
    "Lake Erie":       f"{BASE}/monthly-lake-erie-levels-1921-19.csv",
    "Milk Production": f"{BASE}/monthly-milk-production-pounds-p.csv",
}
NOISE_LAMBDAS = [0.0, 0.005, 0.05, 0.10]
NOISE_LABELS  = {0.0:"0%", 0.005:"0.5%", 0.05:"5%", 0.10:"10%"}

# ═══════════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════════
def load_raw(path):
    return pd.read_csv(path, header=0, parse_dates=[0], index_col=0).values.flatten().astype(float)

def add_noise(raw, lam):
    if lam <= 0: return raw.copy()
    rng = np.random.RandomState(42)
    return raw + rng.normal(0, lam * np.std(raw), size=raw.shape)

def difference(data):
    return np.array([data[i] - data[i-1] for i in range(1, len(data))])

def to_supervised(data, lag=2):
    df = pd.DataFrame(data)
    cols = [df.shift(i) for i in range(1, lag+1)]
    cols.append(df)
    return pd.concat(cols, axis=1).fillna(0).values

def gmf_np(x, c, s):
    return np.exp(-(x - c)**2 / (2 * s**2))

def fuzzy_np(xt, xm1, s):
    ml_t=gmf_np(xt,-1,s); mh_t=gmf_np(xt,1,s)
    ml_m=gmf_np(xm1,-1,s); mh_m=gmf_np(xm1,1,s)
    w1=ml_t*ml_m; w2=ml_t*mh_m; w3=mh_t*ml_m; w4=mh_t*mh_m
    y1=.5*xt+.5*xm1; y2=.7*xt+.3*xm1-.1; y3=.3*xt+.7*xm1+.1; y4=.5*xt+.5*xm1
    return (w1*y1+w2*y2+w3*y3+w4*y4)/(w1+w2+w3+w4+1e-8)

def prep_pipeline(raw_vals, ts, sigma):
    diff = difference(raw_vals)
    sup = to_supervised(diff, 2)
    train, test = sup[:-ts], sup[-ts:]
    sc = MinMaxScaler(feature_range=(-1,1)); sc.fit(train)
    tr_sc, te_sc = sc.transform(train), sc.transform(test)
    def mk_fuzzy(scaled):
        Xr=scaled[:,:-1]; y=scaled[:,-1]
        Xf=np.zeros((Xr.shape[0],2))
        for i in range(Xr.shape[0]):
            Xf[i,0]=Xr[i,1]; Xf[i,1]=fuzzy_np(Xr[i,1],Xr[i,0],sigma)
        return Xf.reshape(-1,1,2).astype('float32'), y.astype('float32')
    X_tr, y_tr = mk_fuzzy(tr_sc)
    return train, test, tr_sc, te_sc, sc, X_tr, y_tr

# ═══════════════════════════════════════════════════════════════
# CELL: LSTM-SNP (for Feature Augmentation — unmodified cell)
# ═══════════════════════════════════════════════════════════════
@tf.keras.utils.register_keras_serializable(package="EvalFFA")
class LSTMSNPCell(tf.keras.layers.Layer):
    def __init__(self, units, **kw):
        super().__init__(**kw); self.units=units; self.state_size=units; self.output_size=units
        self.act=tf.keras.activations.get('tanh'); self.ract=tf.keras.activations.get('hard_sigmoid')
    def build(self, s):
        d=s[-1]
        self.W=self.add_weight((d,self.units*4),initializer='glorot_uniform',name='W')
        self.U=self.add_weight((self.units,self.units*4),initializer='orthogonal',name='U')
        self.b=self.add_weight((self.units*4,),initializer='zeros',name='b')
    def call(self, x, states):
        u=states[0]; z=tf.matmul(x,self.W)+tf.matmul(u,self.U)+self.b
        r=self.ract(z[:,:self.units]); c=self.ract(z[:,self.units:2*self.units])
        o=self.ract(z[:,2*self.units:3*self.units]); a=self.act(z[:,3*self.units:])
        return o*a, [r*u-c*a]
    def get_config(self): return {**super().get_config(),'units':self.units}

# ═══════════════════════════════════════════════════════════════
# CELL: FuzzyLSTMSNP (for Hybrid — fuzzy gates + fuzzy input)
# ═══════════════════════════════════════════════════════════════
@tf.keras.utils.register_keras_serializable(package="EvalHyb")
class FuzzyLSTMSNPCell(tf.keras.layers.Layer):
    def __init__(self, units, mf_sigma=0.25, **kw):
        super().__init__(**kw); self.units=units; self.mf_sigma=mf_sigma
        self.state_size=units; self.output_size=units; self.act=tf.keras.activations.get('tanh')
    def build(self, s):
        d=s[-1]
        self.W=self.add_weight((d,self.units*4),initializer='glorot_uniform',name='W')
        self.U=self.add_weight((self.units,self.units*4),initializer='orthogonal',name='U')
        self.b=self.add_weight((self.units*4,),initializer='zeros',name='b')
        self.fp={}
        for g in ['r','c','o']:
            self.fp[g]={}
            for ri in range(2):
                self.fp[g][ri]={
                    'a':self.add_weight((self.units,),
                        initializer=tf.keras.initializers.RandomUniform(-0.1,0.1),name=f'fz_{g}_r{ri}_a'),
                    'b':self.add_weight((self.units,),
                        initializer=tf.keras.initializers.RandomUniform(-0.1,0.1),name=f'fz_{g}_r{ri}_b'),
                    'c':self.add_weight((self.units,),initializer='zeros',name=f'fz_{g}_r{ri}_c'),
                }
    def _gmf(self, x, c):
        return tf.exp(-tf.square(x-c)/(2.0*self.mf_sigma**2))
    def _fg(self, z, um, g):
        wl=self._gmf(z,-1.0); wh=self._gmf(z,1.0); p=self.fp[g]
        y0=p[0]['a']*z+p[0]['b']*um+p[0]['c']; y1=p[1]['a']*z+p[1]['b']*um+p[1]['c']
        return tf.clip_by_value((wl*y0+wh*y1)/(wl+wh+1e-8),0.,1.)
    def call(self, x, states):
        u=states[0]; z=tf.matmul(x,self.W)+tf.matmul(u,self.U)+self.b
        z0=z[:,:self.units]; z1=z[:,self.units:2*self.units]
        z2=z[:,2*self.units:3*self.units]; z3=z[:,3*self.units:]
        um=tf.tile(tf.reduce_mean(u,axis=-1,keepdims=True),[1,self.units])
        r=self._fg(z0,um,'r'); c=self._fg(z1,um,'c'); o=self._fg(z2,um,'o')
        a=self.act(z3); return o*a, [r*u-c*a]
    def get_config(self): return {**super().get_config(),'units':self.units,'mf_sigma':self.mf_sigma}

# ═══════════════════════════════════════════════════════════════
# TRAINING + INFERENCE
# ═══════════════════════════════════════════════════════════════
def train_and_predict(cell_type, raw_vals, sigma, n_epochs, ts):
    train, test, tr_sc, te_sc, scaler, X_tr, y_tr = prep_pipeline(raw_vals, ts, sigma)
    n_train = len(X_tr)

    np.random.seed(0); tf.random.set_seed(0)
    tf.keras.backend.clear_session()

    if cell_type == 'ffa':
        cell = LSTMSNPCell(UNITS)
    else:
        cell = FuzzyLSTMSNPCell(UNITS, mf_sigma=sigma)

    rnn = tf.keras.layers.RNN(cell, return_sequences=False, stateful=True)
    inp = tf.keras.Input(batch_shape=(1,1,2))
    out = tf.keras.layers.Dense(1)(rnn(inp))
    model = tf.keras.Model(inp, out)

    # Set consumption gate bias = 1.0
    wts = cell.get_weights(); b=wts[2].copy(); b[UNITS:2*UNITS]=1.0; wts[2]=b; cell.set_weights(wts)

    opt = tf.keras.optimizers.Adam(clipnorm=1.0)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # ── Manual GradientTape training ──
    t0 = time.time()
    for ep in range(n_epochs):
        rnn.reset_states()
        for i in range(n_train):
            with tf.GradientTape() as tape:
                pred = model(X_tr[i:i+1], training=True)
                loss = loss_fn(y_tr[i:i+1], pred)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
        if (ep+1) % 25 == 0 or ep == 0:
            print(f"        ep {ep+1}/{n_epochs} loss={loss.numpy():.6f} ({time.time()-t0:.0f}s)", flush=True)

    # ── Warm-up: forward pass through training data ──
    rnn.reset_states()
    for i in range(len(tr_sc)):
        Xr=tr_sc[i,:-1]
        yf=fuzzy_np(Xr[1],Xr[0],sigma)
        model(np.array([[Xr[1],yf]],dtype='float32').reshape(1,1,2), training=False)

    # ── Test predictions → invert to raw space ──
    preds = []
    for i in range(len(te_sc)):
        Xr=te_sc[i,:-1]
        yf=fuzzy_np(Xr[1],Xr[0],sigma)
        yhat=model(np.array([[Xr[1],yf]],dtype='float32').reshape(1,1,2), training=False).numpy()[0,0]
        row = list(Xr) + [yhat]
        inv = scaler.inverse_transform(np.array(row).reshape(1,-1))[0,-1]
        inv += raw_vals[len(train)+i]
        preds.append(inv)

    actual = raw_vals[-ts:]
    print(f"        done ({time.time()-t0:.0f}s)", flush=True)
    return np.array(preds), actual

# ═══════════════════════════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════════════════════════
def naive_bl(vals, ts):
    return vals[-(ts+1):-1], vals[-ts:]
def seasonal_bl(vals, ts, S=12):
    return vals[-(ts+S):-S], vals[-ts:]

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    print("="*80); print("  MODEL PERFORMANCE EVALUATION MATRIX — Actual Inference"); print("="*80, flush=True)

    ckpt_file = os.path.join(BASE, "eval_checkpoint.json")
    if os.path.exists(ckpt_file):
        with open(ckpt_file, 'r') as f:
            completed_combos = json.load(f)
    else:
        completed_combos = {}

    rows = []
    combo = 0
    total_combos = len(DATASETS) * len(NOISE_LAMBDAS) * 2  # 32

    for ds_name, ds_path in DATASETS.items():
        raw = load_raw(ds_path)
        print(f"\n{'━'*60}\n  {ds_name} (n={len(raw)})\n{'━'*60}", flush=True)

        for lam in NOISE_LAMBDAS:
            nl = NOISE_LABELS[lam]
            noisy = add_noise(raw, lam)

            # Baselines
            nf, na = naive_bl(noisy, TEST_SIZE)
            sf, sa = seasonal_bl(noisy, TEST_SIZE)
            n_rmse=sqrt(mean_squared_error(na,nf)); n_r,_=pearsonr(na,nf)
            s_rmse=sqrt(mean_squared_error(sa,sf)); s_r,_=pearsonr(sa,sf)
            print(f"\n  λ={nl}  Naive: RMSE={n_rmse:.4f} r={n_r:.4f} | Season: RMSE={s_rmse:.4f} r={s_r:.4f}", flush=True)

            # Models
            model_results = {}
            for mt, ml in [('ffa','FFA'), ('hybrid','Hybrid')]:
                combo += 1
                key = f"{ds_name}_{nl}_{ml}"
                print(f"    [{combo}/{total_combos}] {ml} ...", flush=True)
                
                if key in completed_combos:
                    m_rmse, m_r = completed_combos[key]
                    print(f"      → (Loaded from Checkpoint) RMSE={m_rmse:.4f}  r={m_r:.6f}", flush=True)
                else:
                    preds, actual = train_and_predict(mt, noisy, MF_SIGMA, N_EPOCHS, TEST_SIZE)
                    m_rmse=sqrt(mean_squared_error(actual,preds)); m_r,_=pearsonr(actual,preds)
                    print(f"      → RMSE={m_rmse:.4f}  r={m_r:.6f}", flush=True)
                    completed_combos[key] = (m_rmse, m_r)
                    with open(ckpt_file, 'w') as f:
                        json.dump(completed_combos, f)
                
                model_results[ml] = (m_rmse, m_r)

            fr, fpr = model_results['FFA']
            hr, hpr = model_results['Hybrid']

            rows.append({
                "ds":ds_name, "nl":nl,
                "n_rmse":n_rmse, "n_r":n_r, "s_rmse":s_rmse, "s_r":s_r,
                "f_rmse":fr, "f_r":fpr, "h_rmse":hr, "h_r":hpr,
                "f_ss_n":(1-fr/n_rmse)*100, "f_ss_s":(1-fr/s_rmse)*100,
                "h_ss_n":(1-hr/n_rmse)*100, "h_ss_s":(1-hr/s_rmse)*100,
            })

    elapsed = time.time() - t_start
    print(f"\n\nTotal time: {elapsed/60:.1f} min", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # PRINT TABLES
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*155)
    print("  TABLE 1: RMSE AND PEARSON'S r")
    print("="*155)
    print(f"{'Dataset':<18} {'Noise':>5} │ {'Naive RMSE':>11} {'r':>8} │ {'Season RMSE':>12} {'r':>8} │ {'FFA RMSE':>10} {'r':>8} │ {'Hybrid RMSE':>12} {'r':>8}")
    print("─"*155)
    p=""
    for r in rows:
        if r['ds']!=p and p: print("─"*155)
        p=r['ds']
        print(f"{r['ds']:<18} {r['nl']:>5} │ {r['n_rmse']:>11.4f} {r['n_r']:>8.4f} │ "
              f"{r['s_rmse']:>12.4f} {r['s_r']:>8.4f} │ {r['f_rmse']:>10.4f} {r['f_r']:>8.4f} │ "
              f"{r['h_rmse']:>12.4f} {r['h_r']:>8.4f}")
    print("─"*155)

    print("\n" + "="*100)
    print("  TABLE 2: SKILL SCORES (%)  SS = (1 − RMSE_Model/RMSE_Baseline) × 100")
    print("="*100)
    print(f"{'Dataset':<18} {'Noise':>5} │ {'FFA SS_N':>10} {'FFA SS_S':>10} │ {'Hyb SS_N':>10} {'Hyb SS_S':>10}")
    print("─"*100)
    p=""
    for r in rows:
        if r['ds']!=p and p: print("─"*100)
        p=r['ds']
        print(f"{r['ds']:<18} {r['nl']:>5} │ {r['f_ss_n']:>+10.2f} {r['f_ss_s']:>+10.2f} │ "
              f"{r['h_ss_n']:>+10.2f} {r['h_ss_s']:>+10.2f}")
    print("─"*100)

    # ── Markdown ──
    print("\n\n### Manuscript Table:\n")
    print("| Dataset | Noise | Metric | Naive Base (x_t+1) | Seasonal Base (x_t+12) | Fuzzy Feature (σ=0.25) | Hybrid Model | SS_Naive (Fuzz / Hyb) | SS_Seasonal (Fuzz / Hyb) |")
    print("|---------|-------|--------|-------------------|----------------------|----------------------|--------------|----------------------|-------------------------|")
    for r in rows:
        print(f"| {r['ds']} | {r['nl']} | RMSE | {r['n_rmse']:.4f} | {r['s_rmse']:.4f} | {r['f_rmse']:.4f} | {r['h_rmse']:.4f} | {r['f_ss_n']:+.2f} / {r['h_ss_n']:+.2f} | {r['f_ss_s']:+.2f} / {r['h_ss_s']:+.2f} |")
        print(f"| {r['ds']} | {r['nl']} | Pearson r | {r['n_r']:.6f} | {r['s_r']:.6f} | {r['f_r']:.6f} | {r['h_r']:.6f} | — | — |")

    # ── Save CSV ──
    csv_path = os.path.join(BASE, "model_performance_evaluation_matrix.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            "Dataset","Noise","Naive_RMSE","Naive_r","Seasonal_RMSE","Seasonal_r",
            "FFA_RMSE","FFA_r","Hybrid_RMSE","Hybrid_r",
            "FFA_SS_Naive","FFA_SS_Seasonal","Hybrid_SS_Naive","Hybrid_SS_Seasonal"])
        w.writeheader()
        for r in rows:
            w.writerow({"Dataset":r['ds'],"Noise":r['nl'],
                "Naive_RMSE":f"{r['n_rmse']:.6f}","Naive_r":f"{r['n_r']:.6f}",
                "Seasonal_RMSE":f"{r['s_rmse']:.6f}","Seasonal_r":f"{r['s_r']:.6f}",
                "FFA_RMSE":f"{r['f_rmse']:.6f}","FFA_r":f"{r['f_r']:.6f}",
                "Hybrid_RMSE":f"{r['h_rmse']:.6f}","Hybrid_r":f"{r['h_r']:.6f}",
                "FFA_SS_Naive":f"{r['f_ss_n']:.2f}","FFA_SS_Seasonal":f"{r['f_ss_s']:.2f}",
                "Hybrid_SS_Naive":f"{r['h_ss_n']:.2f}","Hybrid_SS_Seasonal":f"{r['h_ss_s']:.2f}"})
    print(f"\n✓ CSV → {csv_path}", flush=True)

    # ── Save Markdown ──
    md_path = os.path.join(BASE, "model_performance_evaluation_matrix.md")
    with open(md_path, 'w') as f:
        f.write("# Model Performance Evaluation Matrix\n\n")
        f.write(f"**Config**: MF σ={MF_SIGMA}, units={UNITS}, epochs={N_EPOCHS}, 1 run\n\n")
        f.write("| Dataset | Noise | Metric | Naive Base (x_t+1) | Seasonal Base (x_t+12) | Fuzzy Feature (σ=0.25) | Hybrid Model | SS_Naive (Fuzz / Hyb) | SS_Seasonal (Fuzz / Hyb) |\n")
        f.write("|---------|-------|--------|-------------------|----------------------|----------------------|--------------|----------------------|-------------------------|\n")
        for r in rows:
            f.write(f"| {r['ds']} | {r['nl']} | RMSE | {r['n_rmse']:.4f} | {r['s_rmse']:.4f} | {r['f_rmse']:.4f} | {r['h_rmse']:.4f} | {r['f_ss_n']:+.2f} / {r['h_ss_n']:+.2f} | {r['f_ss_s']:+.2f} / {r['h_ss_s']:+.2f} |\n")
            f.write(f"| {r['ds']} | {r['nl']} | Pearson r | {r['n_r']:.6f} | {r['s_r']:.6f} | {r['f_r']:.6f} | {r['h_r']:.6f} | — | — |\n")
    print(f"✓ MD  → {md_path}", flush=True)
    print("✓ Done.", flush=True)

if __name__ == '__main__':
    main()
