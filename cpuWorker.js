// cpuWorker.js (module)
self.onmessage = (e) => {
  const { type, iterations } = e.data || {};
  if (type !== 'run') return;
  const t0 = performance.now();
  const res = cpuKernel(iterations >>> 0);
  const t1 = performance.now();
  // Post minimal info to avoid overhead
  self.postMessage({
    iterations,
    durationMs: t1 - t0,
    startTime: t0,
    endTime: t1,
    checksum: res >>> 0,
  });
};

// A mixed integer bitwise kernel with good JIT stability
function cpuKernel(iterations) {
  let a = 0x9e3779b9 | 0;
  let b = 0x7f4a7c15 | 0;
  let c = 0x85ebca6b | 0;
  let sum = 0 | 0;
  for (let i = 0; i < iterations; i++) {
    a = (a + 0x6D2B79F5) | 0;
    b ^= a; b = (b << 13) | (b >>> 19);
    c = (c ^ b) + ((c << 7) | (c >>> 25));
    sum = (sum + (a ^ b ^ c)) | 0;
    // lightweight integer divides replaced by imul for x86/ARM parity
    a = Math.imul(a, 1664525) + 1013904223 | 0;
    b = Math.imul(b, 22695477) + 1 | 0;
  }
  return sum ^ a ^ b ^ c;
}
