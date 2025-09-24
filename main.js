/* LZR Web Benchmark main.js */
const state = {
  env: {},
  results: {
    cpuSingle: null,
    cpuMulti: null,
    gpu: null,
  },
};

// ---------- Environment detection ----------
async function detectEnv() {
  const ua = navigator.userAgent;
  const hasUAData = !!navigator.userAgentData;
  let arch = 'unknown';
  let model = '';
  let platform = navigator.platform || 'unknown';
  let brands = [];
  let bitness = '';
  let wow64 = false;
  let platformVersion = '';
  let browser = '';
  let deviceMemory = (navigator.deviceMemory || '?') + ' GB';
  const cores = (navigator.hardwareConcurrency || '?') + ' 逻辑线程';
  // Try UA-CH high-entropy hints
  if (hasUAData && navigator.userAgentData.getHighEntropyValues) {
    try {
      const info = await navigator.userAgentData.getHighEntropyValues([
        'architecture','model','platform','platformVersion','bitness','wow64','fullVersionList'
      ]);
      arch = info.architecture || arch;
      model = info.model || model;
      platform = info.platform || platform;
      platformVersion = info.platformVersion || '';
      bitness = info.bitness || '';
      wow64 = info.wow64 || false;
      brands = (info.fullVersionList || navigator.userAgentData.brands || []).map(b => `${b.brand} ${b.version}`);
      browser = brands.join(', ');
    } catch (e) {}
  }
  if (!browser) {
    // crude fallback
    browser = ua;
  }
  // Architecture heuristics if still unknown
  if (arch === 'unknown' || !arch) {
    const ual = ua.toLowerCase();
    if (ual.includes('arm') || ual.includes('aarch64')) arch = 'arm';
    else if (ual.includes('x86_64') || ual.includes('win64') || ual.includes('x64') || ual.includes('amd64')) arch = 'x86_64';
    else if (ual.includes('i686') || ual.includes('i386') || ual.includes('x86')) arch = 'x86';
  }
  // GPU Info (prefer WebGPU, fallback to WebGL)
  let gpuApi = 'Unknown';
  let gpuName = 'Unknown';
  let gpuDetail = '';
  if ('gpu' in navigator && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        gpuApi = 'WebGPU';
        // Try new API (may be gated)
        if (adapter.requestAdapterInfo) {
          try {
            const info = await adapter.requestAdapterInfo();
            const fields = ['vendor','architecture','device','description','vendorID','deviceID'];
            gpuName = (info.description || info.device || 'Unknown') + '';
            gpuDetail = fields.map(k => info[k] !== undefined ? `${k}:${info[k]}` : '').filter(Boolean).join(' | ');
          } catch {}
        }
        if (!gpuName || gpuName === 'Unknown') {
          // Some browsers expose adapter.name
          gpuName = adapter.name || 'Unknown WebGPU Adapter';
        }
      }
    } catch {}
  }
  if (!gpuName || gpuName === 'Unknown') {
    // Try WebGL debug renderer
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      if (gl) {
        gpuApi = gpuApi === 'Unknown' ? 'WebGL' : gpuApi;
        const dbg = gl.getExtension('WEBGL_debug_renderer_info');
        if (dbg) {
          const vendor = gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL);
          const renderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);
          gpuName = renderer || vendor || 'Unknown WebGL Renderer';
          gpuDetail = [vendor, renderer].filter(Boolean).join(' | ');
        } else {
          gpuName = gl.getParameter(gl.RENDERER);
        }
      }
    } catch {}
  }
  state.env = {
    architecture: arch || 'unknown',
    model: model || '',
    platform: platform + (platformVersion ? ` ${platformVersion}` : ''),
    browser,
    bitness,
    wow64,
    cores,
    deviceMemory,
    gpuApi,
    gpuName,
    gpuDetail,
    userAgent: ua,
  };
  renderEnv();
}

function tagify(text, cls) {
  const span = document.createElement('span');
  span.className = 'tag ' + (cls || '');
  span.textContent = text;
  return span;
}

function renderEnv() {
  const e = state.env;
  const $ = id => document.getElementById(id);
  const tags = document.getElementById('env-tags');
  tags.innerHTML = '';
  tags.appendChild(tagify(`Arch: ${e.architecture}`));
  if (e.bitness) tags.appendChild(tagify(`${e.bitness}-bit`));
  tags.appendChild(tagify(e.gpuApi, e.gpuApi === 'WebGPU' ? 'ok' : (e.gpuApi === 'WebGL' ? 'warn' : 'err')));
  $('device').textContent = e.model || '未知设备（可能受浏览器隐私策略限制）';
  $('platform').textContent = e.platform;
  $('browser').textContent = e.browser;
  $('cpu').textContent = `${e.architecture} · ${e.cores}`;
  $('gpu').textContent = `${e.gpuName} ${e.gpuDetail ? '(' + e.gpuDetail + ')' : ''}`;
  $('memory').textContent = e.deviceMemory;
}

// ---------- CPU Benchmark (Web Workers) ----------
const OPS_LABEL = 'iter/ms'; // relative score

function createWorker() {
  return new Worker('./cpuWorker.js', { type: 'module' });
}

function once(target, type) {
  return new Promise(resolve => {
    const handler = (ev) => { target.removeEventListener(type, handler); resolve(ev); };
    target.addEventListener(type, handler);
  });
}

async function cpuCalibrate() {
  const w = createWorker();
  const iters = 1_000_000; // quick calibration
  const start = performance.now();
  w.postMessage({ type: 'run', iterations: iters });
  const msg = await new Promise(res => w.onmessage = e => res(e.data));
  w.terminate();
  const elapsed = msg.durationMs;
  const itersPerMs = iters / elapsed;
  return { itersPerMs, sampleMs: elapsed };
}

function fmt(n, digits=1) {
  if (!isFinite(n)) return '—';
  if (n >= 1e9) return (n/1e9).toFixed(digits) + 'G';
  if (n >= 1e6) return (n/1e6).toFixed(digits) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(digits) + 'k';
  return n.toFixed(digits);
}

function computeWindow(records, windowLengthMs) {
  if (!records.length) {
    return {
      totalDuration: 0,
      windowDuration: 0,
      totalValue: 0,
      windowValue: 0,
      start: 0,
      end: 0,
    };
  }
  let minStart = records[0].start;
  let maxEnd = records[0].end;
  let totalValue = 0;
  for (const rec of records) {
    if (rec.start < minStart) minStart = rec.start;
    if (rec.end > maxEnd) maxEnd = rec.end;
    totalValue += rec.value;
  }
  const windowEnd = maxEnd;
  const windowStart = Math.max(minStart, windowEnd - windowLengthMs);
  let windowValue = 0;
  for (const rec of records) {
    const overlapStart = Math.max(rec.start, windowStart);
    const overlapEnd = Math.min(rec.end, windowEnd);
    if (overlapEnd > overlapStart) {
      const recDuration = rec.end - rec.start;
      if (recDuration > 0) {
        const fraction = (overlapEnd - overlapStart) / recDuration;
        windowValue += rec.value * fraction;
      } else {
        windowValue += rec.value;
      }
    }
  }
  const windowDuration = windowEnd - windowStart;
  return {
    totalDuration: maxEnd - minStart,
    windowDuration,
    totalValue,
    windowValue,
    start: minStart,
    end: maxEnd,
  };
}

async function runCpuSingle(totalMs = 15_000, windowMs = 10_000) {
  const meta = document.getElementById('cpu-single-meta');
  meta.textContent = '准备中…';
  const { itersPerMs, sampleMs } = await cpuCalibrate();
  const chunkTargetMs = 200;
  const iterationsPerChunk = Math.max(50_000, Math.floor(itersPerMs * chunkTargetMs));
  const worker = createWorker();
  const records = [];
  let firstStart = null;

  async function runChunk() {
    return new Promise(res => {
      worker.onmessage = e => res(e.data);
      worker.postMessage({ type: 'run', iterations: iterationsPerChunk });
    });
  }

  while (true) {
    const msg = await runChunk();
    records.push({ start: msg.startTime, end: msg.endTime, value: msg.iterations });
    if (firstStart === null || msg.startTime < firstStart) firstStart = msg.startTime;
    if (firstStart !== null && (msg.endTime - firstStart) >= totalMs) break;
  }
  worker.terminate();
  const stats = computeWindow(records, windowMs);
  const effectiveWindowMs = stats.windowDuration > 0 ? stats.windowDuration : stats.totalDuration;
  const score = effectiveWindowMs > 0 ? stats.windowValue / effectiveWindowMs : 0;
  state.results.cpuSingle = {
    score,
    totalIterations: stats.totalValue,
    totalMs: stats.totalDuration,
    windowIterations: stats.windowValue,
    windowMs: effectiveWindowMs,
    itersPerMsCalib: itersPerMs,
    calibMs: sampleMs,
    threads: 1,
  };
  document.getElementById('cpu-single-score').textContent = fmt(score) + ' ' + OPS_LABEL;
  meta.textContent = `总迭代: ${fmt(stats.totalValue, 0)} · 总时长: ${(stats.totalDuration/1000).toFixed(1)} s · 后 ${(effectiveWindowMs/1000).toFixed(1)} s 迭代: ${fmt(stats.windowValue, 0)} · 校准: ${sampleMs.toFixed(1)} ms`;
}

async function runCpuMulti(totalMs = 15_000, windowMs = 10_000) {
  const meta = document.getElementById('cpu-multi-meta');
  meta.textContent = '准备中…';
  const { itersPerMs } = await cpuCalibrate();
  const threads = Math.min((navigator.hardwareConcurrency || 4), 16);
  const chunkTargetMs = 200;
  const perWorker = Math.max(50_000, Math.floor(itersPerMs * chunkTargetMs));
  const workers = Array.from({length: threads}, () => createWorker());
  const records = [];
  const startWall = performance.now();
  let running = threads;
  let stopScheduling = false;

  await new Promise(resolve => {
    workers.forEach((worker) => {
      const schedule = () => {
        if (stopScheduling) return;
        worker.postMessage({ type: 'run', iterations: perWorker });
      };
      worker.onmessage = (e) => {
        const msg = e.data;
        records.push({ start: msg.startTime, end: msg.endTime, value: msg.iterations });
        if (!stopScheduling && (performance.now() - startWall) < totalMs) {
          schedule();
        } else {
          stopScheduling = true;
          worker.terminate();
          running--;
          if (running === 0) resolve();
        }
      };
      schedule();
    });
  });

  const stats = computeWindow(records, windowMs);
  const effectiveWindowMs = stats.windowDuration > 0 ? stats.windowDuration : stats.totalDuration;
  const score = effectiveWindowMs > 0 ? stats.windowValue / effectiveWindowMs : 0;
  const totalIterations = stats.totalValue;
  state.results.cpuMulti = {
    score,
    threads,
    totalIterations,
    totalMs: stats.totalDuration,
    windowIterations: stats.windowValue,
    windowMs: effectiveWindowMs,
  };
  document.getElementById('cpu-multi-score').textContent = fmt(score) + ' ' + OPS_LABEL;
  meta.textContent = `线程: ${threads} · 总迭代: ${fmt(totalIterations,0)} · 总时长: ${(stats.totalDuration/1000).toFixed(1)} s · 后 ${(effectiveWindowMs/1000).toFixed(1)} s 迭代: ${fmt(stats.windowValue,0)}`;
}

// ---------- GPU Benchmark ----------
async function runGpu(totalMs = 15_000, windowMs = 10_000) {
  const apiEl = document.getElementById('gpu-api');
  const scoreEl = document.getElementById('gpu-score');
  const metaEl = document.getElementById('gpu-meta');
  apiEl.textContent = state.env.gpuApi || 'Unknown';

  if (state.env.gpuApi === 'WebGPU' && navigator.gpu) {
    // WebGPU compute test: FMA-heavy loop
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const workgroupSize = 256;
    const groups = 2048; // 524,288 invocations
    const invocations = workgroupSize * groups;

    const shader = /* wgsl */`
      struct Params { iters: u32; };
      @group(0) @binding(0) var<storage, read_write> outBuf: array<f32>;
      @group(0) @binding(1) var<uniform> params: Params;

      @compute @workgroup_size(${workgroupSize})
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        var x = f32(gid.x) * 1e-6 + 1.0;
        var y = 0.0;
        for (var i: u32 = 0u; i < params.iters; i = i + 1u) {
          // 3 FMAs + 1 add per loop (≈7 FLOPs if FMA=2)
          x = fma(x, 1.000001, 0.999999);
          x = fma(x, 0.999997, 1.000003);
          y = y + x;
          y = fma(y, 1.0000001, 0.0000001);
        }
        outBuf[gid.x] = y;
      }`;

    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: shader }), entryPoint: 'main' }
    });

    const outBuf = device.createBuffer({
      size: invocations * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const paramBuf = device.createBuffer({
      size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    async function runWithIters(iters) {
      device.queue.writeBuffer(paramBuf, 0, new Uint32Array([iters]));
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: outBuf } },
          { binding: 1, resource: { buffer: paramBuf } }
        ]
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(groups);
      pass.end();
      const t0 = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const t1 = performance.now();
      return { duration: t1 - t0, start: t0, end: t1 };
    }

    // Calibrate
    let iters = 500;
    let warmup = await runWithIters(iters);
    let dt = warmup.duration;
    if (dt < 1) dt = 1;
    const chunkTargetMs = 200;
    const scale = chunkTargetMs / dt;
    let chunkIters = Math.max(200, Math.floor(iters * scale));
    // One more warmup with adjusted iterations (not recorded)
    warmup = await runWithIters(chunkIters);

    const records = [];
    let firstStart = null;
    while (true) {
      const res = await runWithIters(chunkIters);
      records.push({ start: res.start, end: res.end, value: invocations * chunkIters });
      if (firstStart === null || res.start < firstStart) firstStart = res.start;
      if (firstStart !== null && (res.end - firstStart) >= totalMs) break;
    }

    const stats = computeWindow(records, windowMs);
    const effectiveWindowMs = stats.windowDuration > 0 ? stats.windowDuration : stats.totalDuration;
    const score = effectiveWindowMs > 0 ? stats.windowValue / effectiveWindowMs : 0;
    state.results.gpu = {
      api: 'WebGPU',
      score,
      invocations,
      iterationsPerDispatch: chunkIters,
      dispatches: records.length,
      totalWork: stats.totalValue,
      totalMs: stats.totalDuration,
      windowWork: stats.windowValue,
      windowMs: effectiveWindowMs,
    };
    scoreEl.textContent = fmt(score) + ' iter/ms';
    metaEl.textContent = `dispatches: ${records.length} · 每次iters: ${fmt(chunkIters,0)} · 总时长: ${(stats.totalDuration/1000).toFixed(1)} s · 后 ${(effectiveWindowMs/1000).toFixed(1)} s 工作量: ${fmt(stats.windowValue,0)}`;
  } else {
    // WebGL fallback: heavy fragment loop
    const canvas = document.getElementById('glcanvas');
    const gl = canvas.getContext('webgl');
    if (!gl) {
      scoreEl.textContent = '—';
      metaEl.textContent = 'WebGPU/WebGL 不可用';
      return;
    }
    const vs = `attribute vec2 p;void main(){gl_Position=vec4(p,0.0,1.0);}`;
    const fs = `precision highp float;uniform float T;void main(){vec2 uv=gl_FragCoord.xy/512.0;float x=uv.x+T;float y=uv.y;float s=0.0;for(int i=0;i<2000;i++){x = x*1.000001+0.999999;y = y*0.999997+1.000003;s+=x+y;}gl_FragColor=vec4(fract(s),uv,1.0);}`;
    function compile(type, src){
      const sh = gl.createShader(type); gl.shaderSource(sh, src); gl.compileShader(sh);
      if(!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
      return sh;
    }
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    if(!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    const tLoc = gl.getUniformLocation(prog, "T");
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const totalTargetMs = totalMs;
    const records = [];
    let firstStart = null;
    const pixels = canvas.width * canvas.height;
    const workPerFrame = pixels * 2000;
    const nextFrame = () => new Promise(requestAnimationFrame);
    while (true) {
      await nextFrame();
      const start = performance.now();
      gl.uniform1f(tLoc, records.length / 60);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.finish();
      const end = performance.now();
      records.push({ start, end, value: workPerFrame });
      if (firstStart === null || start < firstStart) firstStart = start;
      if (firstStart !== null && (end - firstStart) >= totalTargetMs) break;
    }
    const stats = computeWindow(records, windowMs);
    const effectiveWindowMs = stats.windowDuration > 0 ? stats.windowDuration : stats.totalDuration;
    const score = effectiveWindowMs > 0 ? stats.windowValue / effectiveWindowMs : 0;
    state.results.gpu = {
      api: 'WebGL',
      score,
      pixels,
      frames: records.length,
      itersPerFrag: 2000,
      totalWork: stats.totalValue,
      totalMs: stats.totalDuration,
      windowWork: stats.windowValue,
      windowMs: effectiveWindowMs,
    };
    scoreEl.textContent = fmt(score) + ' work/ms';
    metaEl.textContent = `frames:${records.length} · 总时长:${(stats.totalDuration/1000).toFixed(1)} s · 后 ${(effectiveWindowMs/1000).toFixed(1)} s 工作量:${fmt(stats.windowValue,0)}`;
  }
}

// ---------- Export ----------
function exportJSON() {
  const payload = {
    timestamp: new Date().toISOString(),
    env: state.env,
    results: state.results,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'lzr-web-benchmark.json';
  a.click();
  URL.revokeObjectURL(url);
}

// ---------- Wire up ----------
window.addEventListener('DOMContentLoaded', async () => {
  await detectEnv();
  document.getElementById('btn-cpu-single').onclick = () => runCpuSingle().catch(err => alert(err.message));
  document.getElementById('btn-cpu-multi').onclick = () => runCpuMulti().catch(err => alert(err.message));
  document.getElementById('btn-gpu').onclick = () => runGpu().catch(err => alert(err.message));
  document.getElementById('btn-run-all').onclick = async () => {
    try {
      await runCpuSingle();
      await runCpuMulti();
      await runGpu();
      document.getElementById('btn-export').classList.add('primary');
    } catch (e) { alert(e.message); }
  };
  document.getElementById('btn-export').onclick = exportJSON;
});
