// QwenImage21Gate — parity gates against the oracle goldens + a generate CLI.
//
//   QwenImage21Gate --sched <goldensDir>
//   QwenImage21Gate --attn-probe
//   QwenImage21Gate --vae <weightsRoot> <goldensDir>
//   QwenImage21Gate --encoder <qwenDir> <goldensDir> [--tokenizer <dir>]
//   QwenImage21Gate --dit <weightsRoot> <goldensDir> [--case name]
//   QwenImage21Gate --generate <weightsRoot> <qwenDir> --prompt "..." [--image p.png]... [--size N]
//                   [--out-res N] [--steps N | --sigmas a,b,c | --turbo] [--seed N] [--cfg S] [--neg "..."]
//                   [--out out.png] [--no-cache] [--fp32-vae] [--keep-encoder] [--vae-root <root>]
// Parity gates run fp32 on the CPU stream (the fleet's regime); generate runs bf16 on the GPU.
// `--turbo` samples on the snapshot's own `sample_sigmas` (Qwen-Image-2.1-Turbo, 8 fixed steps);
// `--vae-root` loads the VAE from another snapshot (the Turbo download carries DiT + configs only —
// its VAE is the base one cast to bf16, so the fp32 base VAE is the one to use).

import Foundation
import MLX
import MLXRandom
import QwenImage21

setbuf(stdout, nil)  // line-by-line logs when redirected to a file

// MARK: - helpers

struct Cmp { let cos: Float; let maxAbs: Float; let relMax: Float; let shape: [Int] }

func compare(_ a: MLXArray, _ b: MLXArray) -> Cmp {
    let af = a.asType(.float32).flattened()
    let bf = b.asType(.float32).flattened()
    precondition(af.size == bf.size, "shape mismatch \(a.shape) vs \(b.shape)")
    let dot = sum(af * bf).item(Float.self)
    let na = sqrt(sum(af * af)).item(Float.self)
    let nb = sqrt(sum(bf * bf)).item(Float.self)
    let diff = abs(af - bf)
    let maxAbs = diff.max().item(Float.self)
    let refMax = abs(bf).max().item(Float.self)
    return Cmp(cos: dot / max(na * nb, 1e-30), maxAbs: maxAbs, relMax: maxAbs / max(refMax, 1e-30), shape: a.shape)
}

nonisolated(unsafe) var failures = 0
func report(_ name: String, _ c: Cmp, cosGate: Float = 0.9999, relGate: Float = 2e-2) {
    let ok = c.cos >= cosGate && c.relMax <= relGate
    if !ok { failures += 1 }
    print(String(format: "  %@ %-34@ cos %.8f  maxAbs %.3e  relMax %.3e  %@",
                 ok ? "PASS" : "FAIL", name, c.cos, c.maxAbs, c.relMax, "\(c.shape)"))
}

func loadJSON(_ url: URL) throws -> [String: Any] {
    try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
}

func arg(_ flag: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: flag), i + 1 < a.count else { return nil }
    return a[i + 1]
}
func args(_ flag: String) -> [String] {
    let a = CommandLine.arguments
    return a.indices.filter { a[$0] == flag && $0 + 1 < a.count }.map { a[$0 + 1] }
}
func has(_ flag: String) -> Bool { CommandLine.arguments.contains(flag) }

/// `--sigmas a,b,c` (a fixed grid, terminal 0 excluded) or `--turbo` (the snapshot's own
/// `sample_sigmas`, as Qwen-Image-2.1-Turbo ships them); nil = the shifted linspace schedule.
func samplingGrid(root: URL) throws -> [Float]? {
    if let s = arg("--sigmas") {
        return s.split(separator: ",").map { Float($0.trimmingCharacters(in: .whitespaces))! }
    }
    if has("--turbo") {
        guard let g = try QwenImage21Scheduler.loadSampleSigmas(snapshot: root) else {
            throw QwenImage21Error.loading("--turbo: no sample_sigmas in \(root.path)/model_index.json")
        }
        print("fixed sampling grid (\(g.count) steps): \(g)")
        return g
    }
    return nil
}

/// `--vae-root <snapshot>` overrides where `vae/` is loaded from (defaults to the weights root).
func vaeRoot(default root: URL) -> URL {
    arg("--vae-root").map { URL(fileURLWithPath: $0) } ?? root
}

func bools(_ x: MLXArray) -> [Bool] { x.asType(.bool).asArray(Bool.self) }
func shapes(_ any: Any?) -> [(Int, Int, Int)] {
    (any as? [[Int]] ?? []).map { ($0[0], $0[1], $0[2]) }
}

// MARK: - gates

func gateSched(_ dir: URL) throws {
    let j = try loadJSON(dir.appendingPathComponent("scheduler.json"))
    for s in j["schedules"] as! [[String: Any]] {
        let steps = s["steps"] as! Int, tokens = s["tokens"] as! Int
        let ref = (s["sigmas"] as! [Double]).map { Float($0) }
        let mu: Float
        let ours: [Float]
        if let fixed = s["fixed"] as? [Double] {
            // a checkpoint grid (Turbo `sample_sigmas`): verbatim + trailing 0, whatever mu says
            mu = Float(s["mu"] as! Double)
            ours = try QwenImage21Scheduler.fixedGrid(fixed.map { Float($0) })
        } else {
            mu = QwenImage21Scheduler.calculateShift(imageSeqLen: tokens)
            ours = QwenImage21Scheduler.sigmas(steps: steps, mu: mu)
        }
        let maxAbs = zip(ours, ref).map { abs($0 - $1) }.max() ?? 0
        let ok = ours.count == ref.count && maxAbs < 2e-6 && abs(mu - Float(s["mu"] as! Double)) < 1e-6
        if !ok { failures += 1 }
        print(String(format: "  %@ sched steps=%d tokens=%d mu=%.5f maxAbs %.2e  first %@ last %@",
                     ok ? "PASS" : "FAIL", steps, tokens, mu, maxAbs, "\(ours.prefix(3))", "\(ours.suffix(3))"))
    }
}

func attnProbe() {
    // MLXFast SDPA `.causal` with Lq < Lk must align the query block to the END of the keys
    // (q_i sees keys ≤ i + (Lk - Lq)) — the assumption behind the segment prefill.
    let (lq, lk, h, d) = (5, 9, 2, 16)
    let q = MLXRandom.normal([1, h, lq, d], key: MLXRandom.key(1))
    let k = MLXRandom.normal([1, h, lk, d], key: MLXRandom.key(2))
    let v = MLXRandom.normal([1, h, lk, d], key: MLXRandom.key(3))
    var m = [Float](repeating: -Float.infinity, count: lq * lk)
    for i in 0..<lq { for j in 0..<lk where j <= i + (lk - lq) { m[i * lk + j] = 0 } }
    let mask = MLXArray(m, [1, 1, lq, lk])
    let a = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.25, mask: .causal)
    let b = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.25, mask: .array(mask))
    report("sdpa causal Lq<Lk alignment", compare(a, b), cosGate: 0.999999, relGate: 1e-5)
    // and the `.causal` square case equals the explicit tril
    let q2 = MLXRandom.normal([1, h, lk, d], key: MLXRandom.key(4))
    var m2 = [Float](repeating: -Float.infinity, count: lk * lk)
    for i in 0..<lk { for j in 0...i { m2[i * lk + j] = 0 } }
    let a2 = MLXFast.scaledDotProductAttention(queries: q2, keys: k, values: v, scale: 0.25, mask: .causal)
    let b2 = MLXFast.scaledDotProductAttention(queries: q2, keys: k, values: v, scale: 0.25, mask: .array(MLXArray(m2, [1, 1, lk, lk])))
    report("sdpa causal square", compare(a2, b2), cosGate: 0.999999, relGate: 1e-5)
}

func gateVAE(root: URL, goldens: URL) throws {
    let vae = try QwenImage21Weights.loadVAE(directory: root.appendingPathComponent("vae"), dtype: .float32)
    print("VAE loaded (fp32)")
    for name in ["vae_img_a", "vae_img_b"] {
        let g = try MLX.loadArrays(url: goldens.appendingPathComponent("\(name).safetensors"))
        let pixels = g["pixels_rgba"]![.newAxis]  // (1,4,1,H,W)
        let raw = vae.encodeRaw(pixels)
        eval(raw)
        report("\(name) encode raw", compare(raw[0], g["latents_raw"]!))
        report("\(name) encode normalized", compare(AutoencoderKLQwenImage21.normalize(raw)[0], g["latents_normalized"]!))
        let dec = vae.decode(g["latents_raw"]![.newAxis])
        eval(dec)
        report("\(name) decode", compare(dec[0], g["decoded_rgba"]!))
        // VAE-input construction from the golden PNG (exact bytes expected)
        let caseName = name == "vae_img_a" ? "edit_1img_img_a" : "edit_2img_img_b"
        let png = try QwenImage21PNG.read(url: goldens.appendingPathComponent("\(caseName)_resized_rgba.png"))
        let ours = MLXArray(png.vaePixelsCHW(), [4, 1, png.height, png.width])
        report("\(name) png->vae pixels", compare(ours, g["pixels_rgba"]!), cosGate: 0.999999, relGate: 1e-6)
    }
    let g = try MLX.loadArrays(url: goldens.appendingPathComponent("vae_random_decode.safetensors"))
    let dec = vae.decode(AutoencoderKLQwenImage21.deNormalize(g["latents_normalized"]![.newAxis]))
    eval(dec)
    report("vae_random_decode", compare(dec[0], g["decoded_rgba"]!))
}

func gateResize(goldens: URL) throws {
    for (orig, resizedName, w, h) in [("img_a", "edit_1img_img_a", 320, 320), ("img_b", "edit_2img_img_b", 288, 384)] {
        let src = try QwenImage21PNG.read(url: goldens.appendingPathComponent("\(orig).png"))
        let ref = try QwenImage21PNG.read(url: goldens.appendingPathComponent("\(resizedName)_resized_rgba.png"))
        let ours = QwenImage21PILResize.resizeRGBA(src, outWidth: w, outHeight: h)
        var maxDiff = 0, nDiff = 0
        for i in 0..<ours.rgba.count {
            let d = abs(Int(ours.rgba[i]) - Int(ref.rgba[i]))
            if d > 0 { nDiff += 1 }
            maxDiff = max(maxDiff, d)
        }
        let ok = maxDiff <= 1 && nDiff < ours.rgba.count / 100
        if !ok { failures += 1 }
        print("  \(ok ? "PASS" : "FAIL") lanczos rgba \(orig) -> \(w)x\(h): maxDiff \(maxDiff) differing bytes \(nDiff)/\(ours.rgba.count)")
    }
}

func gateEncoder(qwenDir: URL, goldens: URL, tokenizerDir: URL?) async throws {
    let enc = try await QwenImage21PromptEncoder.load(qwenDir: qwenDir, tokenizerDir: tokenizerDir, dtype: .float32)
    let meta = try loadJSON(goldens.appendingPathComponent("encoder_meta.json"))
    let refDrop = meta["drop_idx"] as! Int
    print("encoder loaded (fp32); drop_idx ours \(enc.dropIdx) ref \(refDrop)")
    if enc.dropIdx != refDrop { failures += 1 }
    for name in ["encoder_t2i_short", "encoder_t2i_card", "encoder_edit_1img", "encoder_edit_2img"] {
        let g = try MLX.loadArrays(url: goldens.appendingPathComponent("\(name).safetensors"))
        let j = try loadJSON(goldens.appendingPathComponent("\(name).json"))
        let prompt = j["prompt"] as! String
        let sizes = j["image_sizes_wh"] as! [[Int]]
        var images: [QwenImage21RGBAImage] = []
        let imgNames = name == "encoder_edit_1img" ? ["img_a"] : name == "encoder_edit_2img" ? ["img_a", "img_b"] : []
        for (i, n) in imgNames.enumerated() {
            let png = try QwenImage21PNG.read(url: goldens.appendingPathComponent("\(name.replacingOccurrences(of: "encoder_", with: ""))_\(n)_resized_rgba.png"))
            precondition(png.width == sizes[i][0] && png.height == sizes[i][1])
            images.append(png)
        }
        let out = try enc.encode(prompt: prompt, images: images)
        let refIds = g["input_ids"]!.asArray(Int32.self).map { Int($0) }
        let idsOK = out.inputIds == refIds
        if !idsOK { failures += 1 }
        print("  \(idsOK ? "PASS" : "FAIL") \(name) token ids: ours \(out.inputIds.count) ref \(refIds.count)\(idsOK ? "" : " first diff at \(zip(out.inputIds, refIds).enumerated().first { $0.1.0 != $0.1.1 }.map { $0.0 } ?? -1)")")
        let refMask = bools(g["image_pad_mask"]!)
        let maskOK = out.imagePadMask == refMask
        if !maskOK { failures += 1 }
        print("  \(maskOK ? "PASS" : "FAIL") \(name) image_pad_mask (\(out.imagePadMask.filter { $0 }.count) pads)")
        if let pv = g["pixel_values"] {
            // our preprocessing of the golden resized RGBA vs the processor's pixel_values
            var parts: [MLXArray] = []
            for img in images { parts.append(enc.processor.preprocess(rgb: img.compositedOverWhiteRGB(), width: img.width, height: img.height).0) }
            report("\(name) pixel_values", compare(parts.count == 1 ? parts[0] : concatenated(parts, axis: 0), pv), cosGate: 0.99999, relGate: 5e-3)
        }
        eval(out.embeds)
        report("\(name) prompt_embeds (pre-norm)", compare(out.embeds[0], g["prompt_embeds_prenorm"]!), cosGate: 0.999, relGate: 5e-2)
    }
}

func gateDiT(root: URL, goldens: URL, only: String?) throws {
    let tr = try QwenImage21Weights.loadTransformer(directory: root.appendingPathComponent("transformer"), dtype: .float32)
    print("transformer loaded (fp32)")
    for name in ["dit_t2i_short", "dit_edit_1img", "dit_edit_2img"] where only == nil || name == only {
        let g = try MLX.loadArrays(url: goldens.appendingPathComponent("\(name).safetensors"))
        let j = try loadJSON(goldens.appendingPathComponent("\(name).json"))
        let imgShapes = shapes(j["img_shapes"])
        let sigmas = (j["sigmas"] as! [Double]).map { Float($0) }
        let nTarget = j["n_target"] as! Int
        print("== \(name): shapes \(imgShapes) sigmas \(sigmas.prefix(2))")
        let layout = try tr.buildLayout(imgMask: bools(g["img_mask"]!), imgShapes: imgShapes)
        // token metadata
        let refPad = bools(g["image_pad_mask_joint"]!)
        let refTarget = bools(g["target_token_mask"]!)
        let prefixOK = layout.imagePadMask == refPad && layout.prefixLen == (j["prefix_len"] as! Int) && refTarget.filter { $0 }.count == nTarget
        if !prefixOK { failures += 1 }
        print("  \(prefixOK ? "PASS" : "FAIL") layout: joint \(layout.jointLen) prefix \(layout.prefixLen) target \(layout.targetLen) segments \(layout.segments.map { "\($0.isText ? "T" : "I")\($0.start)-\($0.end)" })")
        report("rotary cos", compare(layout.cos, g["rotary_emb.real"]!), cosGate: 0.999999, relGate: 1e-4)
        report("rotary sin", compare(layout.sin, g["rotary_emb.imag"]!), cosGate: 0.999999, relGate: 1e-4)

        let pe = g["prompt_embeds"]![.newAxis]
        let hidden0 = g["hidden_in_step0"]![.newAxis]
        var taps: [Int: MLXArray] = [:]
        tr.blockTap = { i, x in taps[i] = x }
        let cache = QwenImage21KVCache(numLayers: tr.numLayers)
        let out0 = tr(hiddenStates: hidden0, encoderHiddenStates: pe, timestep: MLXArray([sigmas[0]]), layout: layout,
                      kvCache: cache, mode: .extract)
        eval(out0)
        report("temb", compare(taps[-2]!, g["temb"]!), cosGate: 0.999999, relGate: 1e-3)
        report("modulation", compare(taps[-3]!, g["modulation"]!), cosGate: 0.999999, relGate: 1e-3)
        for i in [0, 1, 7, 15, 31] {
            report(String(format: "block_%02d", i), compare(taps[i]![0], g[String(format: "block_%02d", i)]!), cosGate: 0.9999, relGate: 2e-2)
        }
        report("out_step0_joint", compare(out0[0], g["out_step0_joint"]!), cosGate: 0.9999, relGate: 2e-2)
        report("out_step0 target rows", compare(out0[0, (out0.dim(1) - nTarget)...], g["out_step0_joint"]![(out0.dim(1) - nTarget)...]), cosGate: 0.9999, relGate: 2e-2)
        report("kv cache L0 k", compare(cache.layers[0].k![0], g["kv_cache_layer0_k"]!), cosGate: 0.99999, relGate: 5e-3)
        report("kv cache L0 v", compare(cache.layers[0].v![0], g["kv_cache_layer0_v"]!), cosGate: 0.99999, relGate: 5e-3)
        report("kv cache L31 k", compare(cache.layers[31].k![0], g["kv_cache_layer31_k"]!), cosGate: 0.9999, relGate: 2e-2)
        tr.blockTap = nil
        // step 1: cached vs uncached with the golden step-1 latents
        let lat1 = g["latents_step1"]![.newAxis]
        let condLen = hidden0.dim(1) - nTarget
        let hidden1 = condLen > 0 ? concatenated([hidden0[0..., ..<condLen], lat1], axis: 1) : lat1
        let out1c = tr(hiddenStates: hidden1, encoderHiddenStates: pe, timestep: MLXArray([sigmas[1]]), layout: layout,
                       kvCache: cache, mode: .cached)
        eval(out1c)
        report("out_step1_cached", compare(out1c[0], g["out_step1_cached"]!), cosGate: 0.9999, relGate: 2e-2)
        let out1u = tr(hiddenStates: hidden1, encoderHiddenStates: pe, timestep: MLXArray([sigmas[1]]), layout: layout, mode: .none)
        eval(out1u)
        report("out_step1_uncached_joint", compare(out1u[0], g["out_step1_uncached_joint"]!), cosGate: 0.9999, relGate: 2e-2)
        report("cached vs uncached (ours)", compare(out1c[0], out1u[0, (out1u.dim(1) - nTarget)...]), cosGate: 0.9999, relGate: 2e-2)
    }
}

/// Production-scale gate: the 1024² edit layout golden (dit_large_edit_1024), run at `dtype` on the
/// current default device. bf16 on the GPU is the production regime — this is where a scale-only
/// divergence (kernel windows, long-graph dispatch) shows up while the 320² fp32 gates stay exact.
func gateDiTLarge(root: URL, goldens: URL, dtype: DType) throws {
    let tr = try QwenImage21Weights.loadTransformer(directory: root.appendingPathComponent("transformer"), dtype: dtype)
    print("transformer loaded (\(dtype))")
    let g = try MLX.loadArrays(url: goldens.appendingPathComponent("dit_large_edit_1024.safetensors"))
    let j = try loadJSON(goldens.appendingPathComponent("dit_large_edit_1024.json"))
    let imgShapes = shapes(j["img_shapes"])
    let sigmas = (j["sigmas"] as! [Double]).map { Float($0) }
    let nTarget = j["n_target"] as! Int
    let layout = try tr.buildLayout(imgMask: bools(g["img_mask"]!), imgShapes: imgShapes)
    print("== dit_large_edit_1024: joint \(layout.jointLen) prefix \(layout.prefixLen) target \(layout.targetLen) segments \(layout.segments.map { "\($0.isText ? "T" : "I")\($0.start)-\($0.end)" })")
    let pe = g["prompt_embeds"]![.newAxis].asType(dtype)
    let hidden0 = concatenated([g["cond_latents_packed"]![.newAxis], g["latents_step0"]![.newAxis]], axis: 1).asType(dtype)
    var taps: [Int: MLXArray] = [:]
    tr.blockTap = { i, x in if [0, 15, 31].contains(i) { taps[i] = x } }
    let cache = QwenImage21KVCache(numLayers: tr.numLayers)
    let t0 = Date()
    let out0 = tr(hiddenStates: hidden0, encoderHiddenStates: pe, timestep: MLXArray([sigmas[0]]), layout: layout, kvCache: cache, mode: .extract)
    eval(out0)
    print(String(format: "  extract forward %.1fs", Date().timeIntervalSince(t0)))
    let gate: Float = dtype == .float32 ? 0.9999 : 0.995
    let rel: Float = dtype == .float32 ? 2e-2 : 1.5e-1
    for i in [0, 15, 31] {
        report(String(format: "block_%02d (joint)", i), compare(taps[i]![0], g[String(format: "block_%02d", i)]!), cosGate: gate, relGate: rel)
        report(String(format: "block_%02d (target rows)", i), compare(taps[i]![0, layout.prefixLen...], g[String(format: "block_%02d", i)]![layout.prefixLen...]), cosGate: gate, relGate: rel)
    }
    report("out_step0 target rows", compare(out0[0, layout.prefixLen...], g["out_step0_joint"]![layout.prefixLen...]), cosGate: gate, relGate: rel)
    report("out_step0 prefix rows", compare(out0[0, ..<layout.prefixLen], g["out_step0_joint"]![..<layout.prefixLen]), cosGate: gate, relGate: rel)
    report("kv cache L0 k", compare(cache.layers[0].k![0], g["kv_cache_layer0_k"]!), cosGate: gate, relGate: rel)
    report("kv cache L31 k", compare(cache.layers[31].k![0], g["kv_cache_layer31_k"]!), cosGate: gate, relGate: rel)
    tr.blockTap = nil
    let hidden1 = concatenated([g["cond_latents_packed"]![.newAxis], g["latents_step1"]![.newAxis]], axis: 1).asType(dtype)
    let t1 = Date()
    let out1c = tr(hiddenStates: hidden1, encoderHiddenStates: pe, timestep: MLXArray([sigmas[1]]), layout: layout, kvCache: cache, mode: .cached)
    eval(out1c)
    print(String(format: "  cached forward %.1fs", Date().timeIntervalSince(t1)))
    report("out_step1_cached", compare(out1c[0], g["out_step1_cached"]!), cosGate: gate, relGate: rel)
    if has("--uncached") {
        let out1u = tr(hiddenStates: hidden1, encoderHiddenStates: pe, timestep: MLXArray([sigmas[1]]), layout: layout, mode: .none)
        eval(out1u)
        report("cached vs uncached (ours)", compare(out1c[0], out1u[0, layout.prefixLen...]), cosGate: gate, relGate: rel)
    }
    if let dump = arg("--dump") {
        try MLX.save(arrays: [
            "out_step0_joint": out0[0].asType(.float32), "out_step1_cached": out1c[0].asType(.float32),
            "block_15": taps[15]![0].asType(.float32), "block_31": taps[31]![0].asType(.float32),
        ], url: URL(fileURLWithPath: dump))
        print("dumped to \(dump)")
    }
}

/// Exactness + memory gate for `decodeTiled`: untiled vs tiled over a halo sweep, on one latent.
/// `--random N` uses a seeded N×N normalized latent (no DiT needed); `--latents f` a saved one.
func vaeTileCheck(root: URL) throws {
    let vae = try QwenImage21Weights.loadVAE(directory: root.appendingPathComponent("vae"), dtype: .float32)
    var z: MLXArray
    if let f = arg("--latents") {
        var l = try MLX.loadArrays(url: URL(fileURLWithPath: f))["latents_packed"]!
        if l.ndim == 2 { l = l[.newAxis] }
        let side = Int(Double(l.dim(1)).squareRoot()) * 16
        z = QwenImage21Latents.unpack(l.asType(.float32), pixelHeight: side, pixelWidth: side)
    } else {
        let n = Int(arg("--random") ?? "48")!
        z = MLXRandom.normal([1, 64, 1, n, n], key: MLXRandom.key(7))
    }
    z = AutoencoderKLQwenImage21.deNormalize(z)
    let tiles = Int(arg("--tiles") ?? "2")!
    let halos = (arg("--halos") ?? "0,4,8,10,11,12,16").split(separator: ",").compactMap { Int($0) }
    print("latent \(z.shape)  output \(z.dim(3) * 16)²  tiles \(tiles)×\(tiles)  device \(Device.defaultDevice())")
    Memory.clearCache(); Memory.peakMemory = 0
    let base0 = Memory.activeMemory
    var t = Date()
    let ref = vae.decode(z); eval(ref)
    let refPeak = Memory.peakMemory - base0
    print(String(format: "  untiled                peak +%.2f GB  %.1fs", Double(refPeak) / 1e9, Date().timeIntervalSince(t)))
    let r8 = clip((ref + 1) * 127.5, min: 0, max: 255).round()
    var dump: [String: MLXArray] = ["latents_denorm": z, "untiled": ref]
    for h in halos {
        Memory.clearCache(); Memory.peakMemory = 0
        let b0 = Memory.activeMemory
        t = Date()
        let out = vae.decodeTiled(z, tilesH: tiles, tilesW: tiles, halo: h); eval(out)
        let peak = Memory.peakMemory - b0
        let maxAbs = abs(out - ref).max().item(Float.self)
        let o8 = clip((out + 1) * 127.5, min: 0, max: 255).round()
        let mse = mean((o8 - r8).square()).item(Float.self)
        let psnr = mse == 0 ? Float.infinity : 10 * log10(255 * 255 / mse)
        let exact = maxAbs == 0
        print(String(format: "  tiled halo %2d          peak +%.2f GB  %.1fs  max|Δ| %.3e  PSNR(8-bit) %@%@", h,
                     Double(peak) / 1e9, Date().timeIntervalSince(t), maxAbs,
                     psnr.isInfinite ? "∞" : String(format: "%.2f dB", psnr), exact ? "  EXACT" : ""))
        dump["tiled_h\(h)_t\(tiles)"] = out
    }
    if let path = arg("--dump") {
        try MLX.save(arrays: dump, url: URL(fileURLWithPath: path))
        print("dumped \(dump.keys.sorted()) to \(path)")
    }
}

/// bf16-vs-fp32 VAE decode on the same final latents (GPU): is a bf16 decoder visually lossless?
func vaeDTypeCheck(root: URL, latentsFiles: [String]) throws {
    let v32 = try QwenImage21Weights.loadVAE(directory: root.appendingPathComponent("vae"), dtype: .float32)
    let v16 = try QwenImage21Weights.loadVAE(directory: root.appendingPathComponent("vae"), dtype: .bfloat16)
    for f in latentsFiles {
        var l = try MLX.loadArrays(url: URL(fileURLWithPath: f))["latents_packed"]!
        if l.ndim == 2 { l = l[.newAxis] }
        let side = Int(Double(l.dim(1)).squareRoot()) * 16
        let z = AutoencoderKLQwenImage21.deNormalize(QwenImage21Latents.unpack(l.asType(.float32), pixelHeight: side, pixelWidth: side))
        let a = v32.decode(z)
        let b = v16.decode(z.asType(.bfloat16)).asType(.float32)
        eval(a, b)
        // PSNR on the 8-bit RGB the user actually gets
        let a8 = clip((a + 1) * 127.5, min: 0, max: 255).round()
        let b8 = clip((b + 1) * 127.5, min: 0, max: 255).round()
        let mse = mean((a8[0..., ..<3] - b8[0..., ..<3]).square()).item(Float.self)
        let psnr = 10 * log10(255 * 255 / max(mse, 1e-9))
        let d = abs(a8 - b8)
        let maxDiff = d.max().item(Float.self)
        let over4 = mean((d .> 4).asType(.float32)).item(Float.self) * 100
        let over8 = mean((d .> 8).asType(.float32)).item(Float.self) * 100
        let over16 = mean((d .> 16).asType(.float32)).item(Float.self) * 100
        print(String(format: "  %@  PSNR %.2f dB  max|Δ| %.0f  |Δ|>4: %.3f%%  >8: %.4f%%  >16: %.5f%%",
                     (f as NSString).lastPathComponent, psnr, maxDiff, over4, over8, over16))
    }
}

/// Measured split footprint for the manifest (efficiency contract 1.14): resident floor = the
/// weights after load with the cache cleared; peak = the worst MLX active+cache during a request at
/// the input envelope; activation = peak − floor, declared at +20%.
///
/// Steps are deliberately low (memory is driven by the per-step graph shape and the one decode, not
/// by how many steps run). The per-request Qwen3-VL load is a TRANSIENT inside `generate`, so it
/// lands in the activation term — matching the 2511 package's convention.
/// ⚠ These are MLX-pool numbers, not in-app `phys_footprint` (the BiRefNet ~2.7× lesson).
func memBench(root: URL, qwenDir: URL) async throws {
    let steps = Int(arg("--steps") ?? "2")!
    let grid = try samplingGrid(root: root)
    let t0 = Date()
    let tr = try QwenImage21Weights.loadTransformer(directory: root.appendingPathComponent("transformer"), dtype: .bfloat16)
    let vae = try QwenImage21Weights.loadVAE(directory: vaeRoot(default: root).appendingPathComponent("vae"), dtype: has("--bf16-vae") ? .bfloat16 : .float32)
    eval(tr, vae)
    Memory.clearCache()
    let floor = Memory.activeMemory
    print(String(format: "resident floor (DiT bf16 + VAE %@, post-load, cache cleared): %.2f GB   [load %.1fs]",
                 has("--bf16-vae") ? "bf16" : "fp32", Double(floor) / 1e9, Date().timeIntervalSince(t0)))
    let gen = QwenImage21Generator(
        encoderProvider: { try await QwenImage21PromptEncoder.load(qwenDir: qwenDir, dtype: .bfloat16) },
        transformer: tr, vae: vae, keepEncoderResident: false)
    let photo = arg("--image").map { URL(fileURLWithPath: $0) }
    let ref = try photo.map { try QwenImage21PNG.read(url: $0) }
    // (name, references, width, height, output_resolution)
    var envelopes: [(String, [QwenImage21RGBAImage], Int?, Int?, Int)] = [
        ("T2I 1024²", [], 1024, 1024, 1024), ("T2I 2048²", [], 2048, 2048, 2048)]
    if let ref {
        envelopes.append(("edit 1024², 1 ref", [ref], nil, nil, 1024))
        envelopes.append(("edit 1024², 4 refs", [ref, ref, ref, ref], nil, nil, 1024))
    }
    // `--only-refs N`: measure just one N-reference edit envelope (e.g. the model's max of 10).
    if let n = arg("--only-refs").flatMap(Int.init), let ref {
        envelopes = [("edit 1024², \(n) refs", Array(repeating: ref, count: n), nil, nil, 1024)]
    }
    // `--t2i-only`: the text-to-image sizes the tiled decode opens up.
    if has("--t2i-only") {
        envelopes = [("T2I 1024²", [], 1024, 1024, 1024), ("T2I 2048²", [], 2048, 2048, 2048),
                     ("T2I 2400×1792 (max)", [], 2400, 1792, 2048), ("T2I 2752×1536", [], 2752, 1536, 2048)]
    }
    // NB: never pass a Swift String to a C `%s` — it segfaults (exit 139). Pad in Swift, format with %@.
    func pad(_ s: String, _ n: Int) -> String { s.count >= n ? s : s + String(repeating: " ", count: n - s.count) }
    print(pad("envelope", 22) + "   floor GB    peak GB     act GB   declare GB")
    for (name, images, width, height, res) in envelopes {
        Memory.clearCache()
        Memory.peakMemory = 0
        let t = Date()
        _ = try await gen.generate(
            prompt: "a red fox in fresh snow", images: images, width: width, height: height,
            outputResolution: res, steps: steps, sigmas: grid, seed: 1)
        let peak = Memory.peakMemory
        let act = max(peak - floor, 0)
        print(pad(name, 22) + String(format: " %10.2f %10.2f %10.2f %12.2f   [%.0fs]",
                     Double(floor) / 1e9, Double(peak) / 1e9, Double(act) / 1e9,
                     Double(act) * 1.2 / 1e9, Date().timeIntervalSince(t)))
        Memory.clearCache()
    }
    print(String(format: "\nDeclare: QuantFootprint(.bf16, resident %.0f, peakActivation <worst act above> * 1.2)", Double(floor)))
}

func generate(root: URL, qwenDir: URL) async throws {
    let prompt = arg("--prompt") ?? "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"
    let steps = Int(arg("--steps") ?? "40")!
    let seed = UInt64(arg("--seed") ?? "42")!
    let size = arg("--size").flatMap(Int.init)
    let outRes = Int(arg("--out-res") ?? "1024")!
    let cfg = Float(arg("--cfg") ?? "1")!
    let outPath = arg("--out") ?? "qwen-image-2.1.png"
    let images = try args("--image").map { try QwenImage21PNG.read(url: URL(fileURLWithPath: $0)) }
    // `--latents file.safetensors` injects the reference's initial noise (key `latents_packed`, [1, hw, 64]
    // or [hw, 64]) so a render can be compared numerically with a torch run despite the RNG mismatch.
    var injected: MLXArray? = nil
    if let lp = arg("--latents") {
        let d = try MLX.loadArrays(url: URL(fileURLWithPath: lp))
        var l = d["latents_packed"] ?? d.values.first!
        if l.ndim == 2 { l = l[.newAxis] }
        injected = l
        print("injected noise \(l.shape)")
    }
    let grid = try samplingGrid(root: root)
    let t0 = Date()
    let ditDType: DType = has("--fp32-dit") ? .float32 : .bfloat16
    let tr = try QwenImage21Weights.loadTransformer(directory: root.appendingPathComponent("transformer"), dtype: ditDType)
    let vae = try QwenImage21Weights.loadVAE(directory: vaeRoot(default: root).appendingPathComponent("vae"), dtype: has("--fp32-vae") || has("--fp32-dit") ? .float32 : .bfloat16)
    print(String(format: "loaded DiT %@ + VAE in %.1fs", "\(ditDType)", Date().timeIntervalSince(t0)))
    let gen = QwenImage21Generator(
        encoderProvider: { try await QwenImage21PromptEncoder.load(qwenDir: qwenDir, dtype: .bfloat16) },
        transformer: tr, vae: vae, keepEncoderResident: has("--keep-encoder"))
    let t1 = Date()
    var last = Date()
    let r = try await gen.generate(
        prompt: prompt, images: images, negativePrompt: arg("--neg"), trueCFGScale: cfg,
        width: size, height: size, outputResolution: outRes, steps: steps, sigmas: grid, seed: seed,
        useKVCache: !has("--no-cache"), latents: injected,
        progress: { i, n in
            let now = Date()
            print(String(format: "  step %d/%d  %.2fs  peak %.1f GB", i, n, now.timeIntervalSince(last), Double(Memory.peakMemory) / 1e9))
            last = now
        })
    print(String(format: "generated %dx%d in %.1fs (steps %d, cache %@); peak %.1f GB", r.image.width, r.image.height,
                 Date().timeIntervalSince(t1), r.sigmas.count - 1, has("--no-cache") ? "off" : "on", Double(Memory.peakMemory) / 1e9))
    try QwenImage21PNG.write(r.image, to: URL(fileURLWithPath: outPath))
    print("wrote \(outPath)")
    if let sl = arg("--save-latents") {
        try MLX.save(arrays: ["latents_packed": r.latentsPacked.asType(.float32)], url: URL(fileURLWithPath: sl))
        print("saved final latents to \(sl)")
    }
}

// MARK: - main

let cli = CommandLine.arguments
do {
    if has("--sched") {
        try gateSched(URL(fileURLWithPath: arg("--sched")!))
    } else if has("--attn-probe") {
        attnProbe()
    } else if has("--resize") {
        try gateResize(goldens: URL(fileURLWithPath: arg("--resize")!))
    } else if has("--vae") {
        let root = URL(fileURLWithPath: arg("--vae")!)
        let goldens = URL(fileURLWithPath: cli[cli.firstIndex(of: "--vae")! + 2])
        try Device.withDefaultDevice(.cpu) { try gateVAE(root: root, goldens: goldens) }
    } else if has("--encoder") {
        let q = URL(fileURLWithPath: arg("--encoder")!)
        let goldens = URL(fileURLWithPath: cli[cli.firstIndex(of: "--encoder")! + 2])
        let tok = arg("--tokenizer").map { URL(fileURLWithPath: $0) }
        Device.setDefault(device: Device(.cpu))  // first MLX touch in this process: pins C++ and Swift defaults
        try await gateEncoder(qwenDir: q, goldens: goldens, tokenizerDir: tok)
    } else if has("--dit") {
        let root = URL(fileURLWithPath: arg("--dit")!)
        let goldens = URL(fileURLWithPath: cli[cli.firstIndex(of: "--dit")! + 2])
        if has("--gpu") {
            print("DiT gate on the GPU stream (fp32; expect ~1e-3 GPU accumulation noise vs the CPU goldens)")
            try gateDiT(root: root, goldens: goldens, only: arg("--case"))
        } else {
            try Device.withDefaultDevice(.cpu) { try gateDiT(root: root, goldens: goldens, only: arg("--case")) }
        }
    } else if has("--dit-large") {
        let root = URL(fileURLWithPath: arg("--dit-large")!)
        let goldens = URL(fileURLWithPath: cli[cli.firstIndex(of: "--dit-large")! + 2])
        let dtype: DType = has("--bf16") ? .bfloat16 : .float32
        if has("--cpu") {
            try Device.withDefaultDevice(.cpu) { try gateDiTLarge(root: root, goldens: goldens, dtype: dtype) }
        } else {
            try gateDiTLarge(root: root, goldens: goldens, dtype: dtype)
        }
    } else if has("--vae-tile") {
        let root = URL(fileURLWithPath: arg("--vae-tile")!)
        if has("--cpu") { try Device.withDefaultDevice(.cpu) { try vaeTileCheck(root: root) } }
        else { try vaeTileCheck(root: root) }
    } else if has("--vae-dtype") {
        let root = URL(fileURLWithPath: arg("--vae-dtype")!)
        try vaeDTypeCheck(root: root, latentsFiles: args("--latents"))
    } else if has("--membench") {
        let root = URL(fileURLWithPath: arg("--membench")!)
        let q = URL(fileURLWithPath: cli[cli.firstIndex(of: "--membench")! + 2])
        try await memBench(root: root, qwenDir: q)
    } else if has("--generate") {
        let root = URL(fileURLWithPath: arg("--generate")!)
        let q = URL(fileURLWithPath: cli[cli.firstIndex(of: "--generate")! + 2])
        try await generate(root: root, qwenDir: q)
    } else {
        print("usage: see header"); exit(2)
    }
} catch {
    print("ERROR: \(error)")
    exit(1)
}
print(failures == 0 ? "ALL GATES PASS" : "\(failures) GATE(S) FAILED")
exit(failures == 0 ? 0 : 1)
