# LZR Web Benchmark

一个纯前端的浏览器跑分页面：自动识别运行环境（架构/线程数/GPU），并进行 CPU（单核/多核）、GPU（优先 WebGPU，回退 WebGL）测试。**不上传任何数据**。

## 预览（本地）

直接双击 `index.html` 即可在本地打开进行测试。

> 提示：若浏览器限制了某些信息（如 GPU 型号），页面会优雅降级。

## 部署到 Vercel（静态站点）

1. 新建一个 Git 仓库，把本项目所有文件放进去。
2. 绑定到 Vercel，新建 Project，选择 **Other**（静态），无需构建命令，输出目录为仓库根目录。
3. （可选）为了启用更强的多线程/跨源隔离能力，可在 `vercel.json` 里启用 COOP/COEP 头。如果你的页面会加载第三方资源，可能因 COEP 被拦截，请酌情删除：

```json
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "Cross-Origin-Opener-Policy", "value": "same-origin" },
        { "key": "Cross-Origin-Embedder-Policy", "value": "require-corp" }
      ]
    }
  ]
}
```

> 本项目默认已包含该配置。如遇跨域静态资源被拦截，删除或修改 `vercel.json` 即可。

## 分数解释

- **CPU**：计算单位为 `iter/ms`（每毫秒可完成的迭代次数），其中迭代是稳定的整数位运算内核；单核使用 1 个 Worker，多核并行多个 Worker 并以墙钟时间统计。
- **GPU (WebGPU)**：以 `invocations * iters / ms` 作为相对指标。计算着色器循环中包含多条 FMA 指令，偏向浮点吞吐指标。
- **GPU (WebGL 回退)**：渲染 512×512 片元、每片元固定迭代多次，指标为 `pixels * iterations * frames / ms` 的相对量。

**注意**：在不同浏览器/电源模式下，同一设备分数会有波动；请尽量用同一浏览器做横向对比。

## 常见问题

- **能识别具体 CPU 型号吗？** 浏览器通常不会暴露精确 CPU 型号，仅能通过 UA-CH 推断架构（ARM/x86）与线程数（`hardwareConcurrency`）。
- **能识别具体 GPU 型号吗？** WebGPU 的 `requestAdapterInfo()` 在部分浏览器可提供较详细信息；否则回退到 WebGL 的 `WEBGL_debug_renderer_info` 扩展。
- **为什么多核分数没有线性提升？** 受调度、散热与浏览器策略影响很常见。

## 许可

MIT
